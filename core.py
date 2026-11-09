"""Fitted linear fare models and explicit decision-time quote boundaries."""
import math
import numpy as np

FEATURES = ["route_0", "route_1", "route_2", "lead_scaled", "lead_squared", "departure_sin", "departure_cos", "weekend"]


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def vector(row, leaked=False):
    quoted = finite(row["quoted_at"], "quoted_at")
    if finite(row["feature_observed_at"], "feature_observed_at") > quoted:
        raise ValueError("a required feature was not available at quote time")
    lead = finite(row["lead_days"], "lead_days")
    route = row["route"]
    if isinstance(route, bool) or not isinstance(route, int) or route not in range(4) or not 1 <= lead <= 30:
        raise ValueError("route or lead is outside the declared fixture domain")
    day = finite(row["departure_day"], "departure_day")
    values = [float(route == i) for i in range(3)] + [lead/30, (lead/30)**2,
             math.sin(day*2*math.pi/30), math.cos(day*2*math.pi/30), float(int(day)%7 >= 5)]
    if leaked:
        values.append(finite(row["post_booking_fare"], "post_booking_fare"))
    return values


def fit(rows, alpha, leaked=False):
    if not rows or finite(alpha,"alpha") < 0:
        raise ValueError("nonempty training data and nonnegative alpha required")
    x = np.array([vector(r, leaked) for r in rows])
    y = np.array([finite(r["paid_fare"],"paid_fare") for r in rows])
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(design.shape[1])*np.sqrt(alpha)
    penalty[0,0] = 0
    coef = np.linalg.lstsq(np.vstack([design,penalty]), np.r_[y,np.zeros(len(penalty))], rcond=None)[0]
    return {"schema":"fare-ridge.v1","features":FEATURES+(["post_booking_fare_UNSAFE"] if leaked else []),
            "mean":mean.tolist(),"scale":scale.tolist(),"coef":coef.tolist(),"alpha":alpha,"unsafe_leaked_diagnostic":leaked}


def predict(model, rows):
    if model["schema"] != "fare-ridge.v1":
        raise ValueError("unsupported model schema")
    x = np.array([vector(r, model["unsafe_leaked_diagnostic"]) for r in rows])
    return (np.column_stack([np.ones(len(rows)),(x-np.array(model["mean"]))/np.array(model["scale"])]) @ np.array(model["coef"])).tolist()


def quote(model, row, now):
    now = finite(now,"now")
    start = finite(row["quoted_at"],"quoted_at")
    end = finite(row["expires_at"],"expires_at")
    if end <= start:
        raise ValueError("quote expiry must follow observation")
    if now < start or now >= end:
        raise ValueError("quote snapshot is not valid at request time")
    if model["unsafe_leaked_diagnostic"]:
        raise ValueError("post-booking diagnostic model cannot serve a quote")
    return predict(model,[row])[0]


def generate():
    rng = np.random.default_rng(1602)
    rows = []
    for day in range(160):
        for j in range(15):
            route = int(rng.integers(0,4)); lead = int(rng.integers(1,31))
            departure = day + lead
            fare = 70 + 20*route + .8*(31-lead) + 15*math.sin(departure*2*math.pi/30) + 25*(departure%7>=5) + rng.normal(0,4)
            rows.append({"id":f"d{day:03d}-{j:02d}","day":day,"quoted_at":day*100+j,
                         "expires_at":day*100+j+5,"feature_observed_at":day*100+j-1,
                         "route":route,"lead_days":lead,"departure_day":departure,
                         "paid_fare":round(float(fare),8),"post_booking_fare":round(float(fare+rng.normal(0,.05)),8),
                         "split":"train" if day<100 else "validation" if day<120 else "test"})
    return {"rows":rows,"units":"fictional fare units, not real prices","seed":1602}


def experiment(data):
    rows = data["rows"]
    train = [r for r in rows if r["split"]=="train"]
    val = [r for r in rows if r["split"]=="validation"]
    test = [r for r in rows if r["split"]=="test"]
    def mae(pred, subset): return float(np.mean(np.abs(np.array(pred)-[r['paid_fare'] for r in subset])))
    validation = {str(a):mae(predict(fit(train,a),val),val) for a in [0,1,10]}
    selected = min([0,1,10],key=lambda a:validation[str(a)])
    model = fit(train+val,selected)
    leakage = fit(train+val,selected,leaked=True)
    predictions = predict(model,test)
    leaked_pred = predict(leakage,test)
    baseline = float(np.mean([r['paid_fare'] for r in train+val]))
    raw = []
    for i,row in enumerate(test):
        now = row['quoted_at']+(2 if i%2==0 else 6)
        try:
            value=quote(model,row,now); status="served"
        except ValueError:
            value=None; status="refused_expired"
        raw.append({"id":row['id'],"paid_fare":row['paid_fare'],"ready_prediction":predictions[i],
                    "unsafe_leakage_prediction":leaked_pred[i],"request_time":now,"quote_status":status,"served_prediction":value})
    metrics={"train":len(train),"validation":len(val),"test":len(test),"selected_alpha":selected,
             "validation_mae":validation,"training_mean_test_mae":mae([baseline]*len(test),test),
             "decision_time_test_mae":mae(predictions,test),"unsafe_leakage_test_mae":mae(leaked_pred,test),
             "served_valid_requests":sum(r['quote_status']=='served' for r in raw),
             "refused_expired_requests":sum(r['quote_status']=='refused_expired' for r in raw),
             "numpy":np.__version__,"units":data['units']}
    report=f"""# Fare snapshot decision

Topic: price prediction. Decision: estimate a fictional eventual paid fare using fields available at quote time, then refuse an expired quote.

A random notebook can accidentally include the post-booking fare. My implementation isolates decision-time fields, chooses ridge regularization using earlier validation data, and freezes a fitted model before the test period. The model artifact stores actual fitted coefficients and training-only normalization, not a placeholder.

Across {len(test)} chronological test quotes, mean-control MAE is {metrics['training_mean_test_mae']:.3f} fare units and decision-time model MAE is {metrics['decision_time_test_mae']:.3f}. An intentionally ineligible post-booking diagnostic has MAE {metrics['unsafe_leakage_test_mae']:.3f}. Its better number is leakage, not a candidate release. The quote protocol serves {metrics['served_valid_requests']} valid snapshots and refuses {metrics['refused_expired_requests']} expired requests in the authored request schedule.

The generator is stationary and its price equation is simple. These are fictional units, not airline prices or production accuracy. Quote expiry is a local contract, not a booking guarantee. Next, I would audit actual feature availability and missingness before evaluating on a representative authorized booking dataset.
"""
    return metrics,raw,model,report
