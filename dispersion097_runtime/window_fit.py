"""Whole-study cost/window forecast for the cards that currently fit."""
import math
from dual_coordination import GPUS
from runtime_dispatch import choose_next
def forecast(costs,observations,remaining_seconds):
 assert type(remaining_seconds) in (int,float) and math.isfinite(remaining_seconds)
 eligible={g for g in GPUS if observations.get(g,{}).get('GPU_fits') is True and observations[g].get('host_fits') is True}
 base=dict(eligible_cards=sorted(eligible),remaining_window_seconds=remaining_seconds,safety_multiplier=1.25,additional_report_cleanup_seconds=1200,quality_values_read=False,not_a_completion_guarantee=True)
 if not eligible:return dict(base,all_six_predicted_to_fit=False,reason='waiting_for_shared_capacity',required_seconds=None)
 # choose_next performs the same exact count placement as the actual dispatcher.
 # Only the forecast and registered first key are used; nothing is admitted.
 from preset_scope import cell
 pending=[cell(a,s) for s in (8,9,10) for a in ('theta1','theta5')]
 _,allocation=choose_next(pending,{},next(g for g in GPUS if g in eligible),costs,eligible=eligible)
 makespan=max(allocation['predicted_lane_seconds']);required=1.25*makespan+1200
 return dict(base,all_six_predicted_to_fit=required<=remaining_seconds,reason='all_six_fit_current_eligible_card_forecast' if required<=remaining_seconds else 'current_cards_cannot_finish_all_six_in_window',required_seconds=required,forecast=allocation)
