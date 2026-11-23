# Grid2Op observation, action and reward reference

Network: `l2rpn_icaps_2021_large_train`

## Dimensions and conventions

| Entity           |                  Count |
| :--------------- | ---------------------: |
| Lines            |                     59 |
| Substations      |                     36 |
| Generators       |                     22 |
| Loads            |                     37 |
| Topology entries | 177 = 2 × 59 + 22 + 37 |
| Shunts           |                      6 |
| Storage units    |                      0 |
| Alarm areas      |                      3 |
| Alertable lines  |                      0 |

## Shared observations

Definitions follow the [official observation API](https://grid2op.readthedocs.io/en/latest/user/observation.html), checked against the installed `ObservationICAPS2021` and observation implementation. A comprehensive tutorial is provided in the [getting_started notebook](https://github.com/Grid2op/grid2op/blob/07012c4f19cac81662df1df54122568c64d8fe09/getting_started/02_Observation.ipynb).

| Field                            | JSON shape | Value range                                     | Short description                                                              |
| :------------------------------- | :--------- | :---------------------------------------------- | :----------------------------------------------------------------------------- |
| `year`                           | `(1,)`     | Positive calendar integer                       | Simulation year.                                                               |
| `month`                          | `(1,)`     | 1–12                                            | Simulation month.                                                              |
| `day`                            | `(1,)`     | 1–31; valid calendar date                       | Simulation day of month.                                                       |
| `hour_of_day`                    | `(1,)`     | 0–23                                            | Simulation hour.                                                               |
| `minute_of_hour`                 | `(1,)`     | 0–59                                            | Simulation minute.                                                             |
| `day_of_week`                    | `(1,)`     | 0–6; Monday = 0                                 | Simulation weekday.                                                            |
| `gen_p`                          | `(22,)`    | Real; equipment/scenario dependent              | Generator active power (MW); loads can include signed interchanges.            |
| `gen_q`                          | `(22,)`    | Real; equipment dependent                       | Generator reactive power (MVAr).                                               |
| `gen_v`                          | `(22,)`    | ≥ 0; equipment dependent                        | Generator voltage magnitude (kV).                                              |
| `load_p`                         | `(37,)`    | Real; equipment/scenario dependent              | Load active power (MW); loads can include signed interchanges.                 |
| `load_q`                         | `(37,)`    | Real; equipment dependent                       | Load reactive power (MVAr).                                                    |
| `load_v`                         | `(37,)`    | ≥ 0; equipment dependent                        | Load voltage magnitude (kV).                                                   |
| `p_or`                           | `(59,)`    | Real; signed                                    | Active power at line origin (MW).                                              |
| `q_or`                           | `(59,)`    | Real; signed                                    | Reactive power at line origin (MVAr).                                          |
| `v_or`                           | `(59,)`    | ≥ 0                                             | Voltage magnitude at line origin (kV).                                         |
| `a_or`                           | `(59,)`    | ≥ 0                                             | Current magnitude at line origin (A).                                          |
| `p_ex`                           | `(59,)`    | Real; signed                                    | Active power at line extremity (MW).                                           |
| `q_ex`                           | `(59,)`    | Real; signed                                    | Reactive power at line extremity (MVAr).                                       |
| `v_ex`                           | `(59,)`    | ≥ 0                                             | Voltage magnitude at line extremity (kV).                                      |
| `a_ex`                           | `(59,)`    | ≥ 0                                             | Current magnitude at line extremity (A).                                       |
| `rho`                            | `(59,)`    | ≥ 0; can exceed 1                               | Line current / thermal limit; 1 = 100% loading.                                |
| `line_status`                    | `(59,)`    | Boolean                                         | True if a line is connected.                                                   |
| `timestep_overflow`              | `(59,)`    | Integer ≥ 0                                     | Consecutive overloaded timesteps per line.                                     |
| `topo_vect`                      | `(177,)`   | {-1, 1, 2}                                      | Equipment-terminal busbar; -1 = disconnected.                                  |
| `time_before_cooldown_line`      | `(59,)`    | Integer ≥ 0                                     | Steps until line switching is available.                                       |
| `time_before_cooldown_sub`       | `(36,)`    | Integer ≥ 0                                     | Steps until substation switching is available.                                 |
| `time_next_maintenance`          | `(59,)`    | -1 or integer ≥ 0                               | Steps until maintenance; -1 = none, 0 = ongoing.                               |
| `duration_next_maintenance`      | `(59,)`    | Integer ≥ 0                                     | Upcoming/remaining maintenance duration (steps).                               |
| `target_dispatch`                | `(22,)`    | Real; generator dependent                       | Requested redispatch offset from baseline (MW).                                |
| `actual_dispatch`                | `(22,)`    | Real; generator dependent                       | Realized redispatch offset from baseline (MW).                                 |
| `storage_charge`                 | `(0,)`     | Empty here; otherwise [Emin, Emax]              | Stored energy per storage unit (MWh).                                          |
| `storage_power_target`           | `(0,)`     | Empty here; otherwise [-Pprod,max, Pabsorb,max] | Requested storage power (MW); positive = charging.                             |
| `storage_power`                  | `(0,)`     | Empty here; otherwise equipment dependent       | Realized storage power (MW); positive = charging.                              |
| `gen_p_before_curtail`           | `(22,)`    | ≥ 0; generator dependent                        | Renewable output before curtailment (MW); zero for non-curtailable generators. |
| `curtailment`                    | `(22,)`    | [0, 1]                                          | Curtailed power divided by generator Pmax; not MW.                             |
| `curtailment_limit`              | `(22,)`    | [0, 1]                                          | Requested production cap / Pmax; 1 = unrestricted.                             |
| `is_alarm_illegal`               | `(1,)`     | Boolean                                         | Whether the last alarm attempt violated alarm rules.                           |
| `time_since_last_alarm`          | `(1,)`     | -1 or integer ≥ 0                               | Steps since last successful alarm; -1 = none.                                  |
| `last_alarm`                     | `(3,)`     | -1 or integer in [0, current_step]              | Last successful alarm timestep per area; -1 = none.                            |
| `attention_budget`               | `(1,)`     | [0, 3] in this configuration                    | Remaining simulator alarm budget; not human attention.                         |
| `was_alarm_used_after_game_over` | `(1,)`     | Boolean                                         | Whether an alarm contributed to terminal scoring.                              |
| `_shunt_p`                       | `(6,)`     | Real; zero in this session                      | Shunt active power (MW).                                                       |
| `_shunt_q`                       | `(6,)`     | Real; zero in this session                      | Shunt reactive power (MVAr).                                                   |
| `_shunt_v`                       | `(6,)`     | ≥ 0; zero in this session                       | Shunt voltage magnitude (kV).                                                  |
| `_shunt_bus`                     | `(6,)`     | {-1, 1, 2}; all -1 in this session              | Shunt busbar; -1 = disconnected.                                               |
| `current_step`                   | `(1,)`     | Integer ≥ 0                                     | Current simulation step.                                                       |
| `max_step`                       | `(1,)`     | Integer ≥ 0 for finite episodes                 | Episode step limit; horizon-dependent.                                         |
| `delta_time`                     | `(1,)`     | > 0; 5 here                                     | Interval between past and current steps (minutes).                             |
| `gen_margin_up`                  | `(22,)`    | ≥ 0 in feasible states; generator dependent     | Available generation increase at next step (MW).                               |
| `gen_margin_down`                | `(22,)`    | ≥ 0 in feasible states; generator dependent     | Available generation decrease at next step (MW).                               |
| `_thermal_limit`                 | `(59,)`    | > 0 for configured lines                        | Per-line current limit (A).                                                    |
| `support_theta`                  | `(1,)`     | Boolean                                         | Whether voltage angles are supported.                                          |
| `theta_or`                       | `(59,)`    | Real; reference-dependent                       | Line-origin voltage angle (degrees); requires support_theta.                   |
| `theta_ex`                       | `(59,)`    | Real; reference-dependent                       | Line-extremity voltage angle (degrees); requires support_theta.                |
| `load_theta`                     | `(37,)`    | Real; reference-dependent                       | Load voltage angle (degrees); requires support_theta.                          |
| `gen_theta`                      | `(22,)`    | Real; reference-dependent                       | Generator voltage angle (degrees); requires support_theta.                     |
| `storage_theta`                  | `(0,)`     | Real; reference-dependent; empty here           | Storage voltage angle (degrees); requires support_theta.                       |

## Additional fields in the historical export

These eight fields are absent from this notebook’s `ObservationICAPS2021.to_json()` output. Empty `(0,)` fields carry no values in this session; their domains below describe the general feature when supported. Alarm areas and line alerts are different mechanisms.

| Field                         | JSON shape | Value range when populated | Short description                                                       |
| :---------------------------- | :--------- | :------------------------- | :---------------------------------------------------------------------- |
| `active_alert`                | `(0,)`     | Boolean                    | Whether an alert is active for each alertable line.                     |
| `alert_duration`              | `(0,)`     | Integer ≥ 0                | Consecutive active-alert duration (steps).                              |
| `attack_under_alert`          | `(0,)`     | {-1, 0, 1}                 | Recent attack: -1 without prior alert, +1 with alert, 0 none in window. |
| `curtailment_limit_effective` | `(22,)`    | [0, 1]                     | Production cap / Pmax actually imposed by environment.                  |
| `time_since_last_alert`       | `(0,)`     | -1 or integer ≥ 0          | Steps since last alert; -1 = never.                                     |
| `time_since_last_attack`      | `(0,)`     | -1 or integer ≥ 0          | Steps since last attack began; -1 = never.                              |
| `total_number_of_alert`       | `(0,)`     | Integer ≥ 0                | Cumulative alert counter; empty in this export.                         |
| `was_alert_used_after_attack` | `(0,)`     | {-1, 0, 1}                 | Alert evaluation: incorrect, none, or correct, respectively.            |

## Actions supported by this environment

Action list can be find here: [Action API](https://grid2op.readthedocs.io/en/latest/user/action.html) and a detailed examples are provided in the [action tutorial](https://github.com/Grid2op/grid2op/blob/07012c4f19cac81662df1df54122568c64d8fe09/getting_started/03_Action.ipynb).

| Action               | Dense shape             | Values / bounds                                                                 | Short description                                                          |
| :------------------- | :---------------------- | :------------------------------------------------------------------------------ | :------------------------------------------------------------------------- |
| No-op (`{}`)         | No standalone component | Neutral values in every component                                               | Request no control change; time and external conditions still advance.     |
| `set_line_status`    | `(59,)`                 | `-1` disconnect; `0` unchanged; `+1` reconnect                                  | Set line connection status.                                                |
| `change_line_status` | `(59,)`                 | Boolean; true = toggle                                                          | Toggle selected line statuses.                                             |
| `set_bus`            | `(177,)`                | `-1` disconnect; `0` unchanged; `1`/`2` select bus                              | Assign equipment terminals to busbars.                                     |
| `change_bus`         | `(177,)`                | Boolean; true = switch between bus 1 and 2                                      | Toggle selected terminal busbars.                                          |
| `redispatch`         | `(22,)`                 | Per generator: `[-gen_max_ramp_down[i], +gen_max_ramp_up[i]]` MW; `0` unchanged | Request an incremental change to dispatch; only redispatchable generators. |
| `curtail`            | `(22,)`                 | `-1` unchanged; `[0,1]` fraction of Pmax                                        | Cap renewable generation; `1` removes the cap.                             |
| `raise_alarm`        | `(3,)`                  | Boolean per area; true = raise alarm                                            | Signal risk in selected alarm areas, subject to budget.                    |

Examples of sparse input: `{"set_line_status": [(0, 1)]}`, `{"set_bus": {"generators_id": [(0, 2)]}}`, `{"redispatch": [(0, 1.0)]}`. These illustrate syntax, not guaranteed legal actions. A substation topology vector has shape `(env.sub_info[sub_id],)`.
Action type `set` and `change` can be used alternatively: `set` imposes a value to an entity, even the same value of the previous step; `change` just invert the value of an entity like a switch.

## Reward functions

Every reward call returns one **scalar**. The current environment configuration selects **`AlarmReward`**. This greedy agent instead ranks simulated actions by minimum maximum `rho`, without optimizing the returned environment reward.

The public [Grid2Op reward catalogue](https://grid2op.readthedocs.io/en/latest/user/reward.html) provides these choices. Unless stated, numerical scaling depends on the reward implementation and configuration.

| Reward                   | Very short description                                                                                                                                                                                                                                                                                                                      |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `AlarmReward`            | Terminal survival/alarm timing-and-location score; 0 during operation, +1 if episode ends successfully, -1 if alarms are not identified.                                                                                                                                                                                                    |
| `AlertReward`            | Delayed score for line alerts around opponent attacks; requires alert support (not here). Values {+1,+2,-1,-10}.                                                                                                                                                                                                                            |
| `BridgeReward`           | Computes a penalty based on how many bridges are present in the network. A bridge is an edge that if removed will cause the graph to be disconnected.                                                                                                                                                                                       |
| `CloseToOverflowReward`  | Finds all lines close to overflowing. Returns max reward when there is no overflow, min reward if more than one line is close to overflow and the mean between max and min reward if just one line is close to overflow.                                                                                                                    |
| `ConstantReward`         | Always zero.                                                                                                                                                                                                                                                                                                                                |
| `DistanceReward`         | Computes a penalty based on the distance of the current grid to the grid at time 0 where everything is connected to bus 1.                                                                                                                                                                                                                  |
| `EconomicReward`         | Computes the marginal cost of the powergrid. This class also ensures that: the reward is positive if there is no game over, no error etc.; the reward is inversely proportional to the cost of the grid (the higher the reward, the lower the economic cost).                                                                               |
| `EpisodeDurationReward`  | Will always be 0., unless at the end of an episode where it will return the number of steps made by the agent divided by the total number of steps possible in the episode.                                                                                                                                                                 |
| `FlatReward`             | This reward return a fixed number (if there are not error) or 0 if there is an error.                                                                                                                                                                                                                                                       |
| `GameplayReward`         | Strictly computed based on the Game status. It yields a negative reward in case of game over. A half negative reward on rules infringement. Otherwise the reward is positive.                                                                                                                                                               |
| `IncreasingFlatReward`   | This reward just counts the number of timestep the agent has successfully manage to perform. It adds a constant reward for each time step successfully handled.                                                                                                                                                                             |
| `L2RPNReward`            | This rewards makes the sum of the “squared margin” on each powerline. The margin is defined, for each powerline as: margin of a powerline = (thermal limit - flow in amps) / thermal limit (if flow in amps <= thermal limit) else margin of a powerline = 0. This rewards is then: sum (margin of this powerline) ^ 2, for each powerline. |
| `LinesCapacityReward`    | Reward based on lines capacity usage: returns max reward if no current is flowing in the lines; returns min reward if all lines are used at max capacity                                                                                                                                                                                    |
| `LinesReconnectedReward` | This reward computes a penalty based on the number of powerline that could have been reconnected (cooldown at 0.) but are still disconnected.                                                                                                                                                                                               |
| `N1Reward`               | This class implements a reward that is inspired by the “n-1” criterion widely used in power system. More specifically it returns the maximum flows (on all the powerlines) after a given (as input) a powerline has been disconnected.                                                                                                      |
| `RedispReward`           | This reward can be used for environments where redispatching is available. It assigns a cost to redispatching action and penalizes with the losses. This is the closest reward to the score used for the l2RPN competitions.                                                                                                                |
| `CombinedReward`         | Weighted sum of component rewards.                                                                                                                                                                                                                                                                                                          |
| `CombinedScaledReward`   | Rescaled weighted sum; default output range [-0.5,0.5].                                                                                                                                                                                                                                                                                     |

`N1Reward` is a loading metric despite its name: its implementation returns -1 on power-flow divergence. Do not maximize the raw value or treat divergence as a desirable low-loading state. `L2RPNSandBoxScore` and `L2RPNWCCI2022ScoreFun` are internal competition cost scores, not ordinary rewards to maximize. `BaseReward` is the customization interface; `RewardHelper` is infrastructure.

## Candidate reward features for inverse RL

These are **proposed feature definitions**, not additional built-in rewards. A model can learn weights for them; positive costs below usually receive negative weights. Evaluate electrical outcome features on `s_next`, with terminal/failure flags retained.

| Feature              | Shape  | Definition / range                                         | Meaning                                                         |
| :------------------- | :----- | :--------------------------------------------------------- | :-------------------------------------------------------------- |
| Maximum loading      | `()`   | `max(rho)`; ≥ 0                                            | Worst congestion.                                               |
| Overload count       | `()`   | `sum(rho > 1)`; 0–59                                       | Number of overloaded lines.                                     |
| Overload severity    | `()`   | `sum(max(rho - 1, 0))`; ≥ 0                                | Total excess loading.                                           |
| Disconnection count  | `()`   | `sum(~line_status)`; 0–59                                  | Disconnected lines.                                             |
| Topology distance    | `()`   | `sum(topo_vect != reference)`; 0–177                       | Terminal differences from an explicit reference.                |
| Redispatch magnitude | `()`   | `sum(abs(actual_dispatch))` MW; ≥ 0                        | Realized dispatch effort; not automatically the exporter’s KPI. |
| Curtailed power      | `()`   | `sum(curtailment * gen_pmax)` MW; ≥ 0                      | Renewable power withheld.                                       |
| Switching effort     | `()`   | Count requested changed terminals/statuses relative to `s` | Action effort; distinguish requested from executed changes.     |
| Invalid action       | `()`   | Boolean from `info` legality/ambiguity flags               | Failed control request.                                         |
| Failure / completion | `(2,)` | Separate Boolean indicators                                | Distinguish blackout from successful episode completion.        |

Store `(state, requested_action, next_state, reward, done, info, scenario_id, timestep)` and the reward definition. Inverse RL can infer reward from demonstrations without known rewards. Do not infer executed actions solely from state differences: automatic trips and external events also change state.

## Notes for the supplied real-world session

- The six event records contain two distinct observation snapshots. Traces 0–3 have zero electrical values, disconnected lines and topology -1, consistent with terminal-state encoding; zero loading there is not a good outcome. Traces 4–5 duplicate one plausible operating state.
- Simulation dates and session logging dates differ. Preserve both; do not assume event order is trajectory order.
- Anticipation-event `metadata.flux` is contingency loading in percent, not current `rho`. Metadata KPIs are not automatically the environment reward or the proposed reward features.
- Match asset IDs/order before comparing vectors. The reference topology need not be all bus 1.
