# Fare snapshot decision

Topic: price prediction. Decision: estimate a fictional eventual paid fare using fields available at quote time, then refuse an expired quote.

A random notebook can accidentally include the post-booking fare. My implementation isolates decision-time fields, chooses ridge regularization using earlier validation data, and freezes a fitted model before the test period. The model artifact stores actual fitted coefficients and training-only normalization, not a placeholder.

Across 600 chronological test quotes, mean-control MAE is 24.067 fare units and decision-time model MAE is 3.325. An intentionally ineligible post-booking diagnostic has MAE 0.041. Its better number is leakage, not a candidate release. The quote protocol serves 300 valid snapshots and refuses 300 expired requests in the authored request schedule.

The generator is stationary and its price equation is simple. These are fictional units, not airline prices or production accuracy. Quote expiry is a local contract, not a booking guarantee. Next, I would audit actual feature availability and missingness before evaluating on a representative authorized booking dataset.
