# Canonical "full" v26r channel flags (source this).  CUST = customers per step.
# Loyalty pool ~ one day of returning requests so each active rider requests about once a day.
CUST=${CUST:-100}
V26R_FULL_FLAGS="--spillover_price_aware --supply_model v2 --driver_pay_basis fare --loyalty_model v2 --loyalty_pool_size $(( CUST * 6 * 6 / 10 )) --persistence_features"
V26R_FULL_RIVAL="adaptive_best_response_memory"
