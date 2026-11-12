#!/bin/zsh
# Table III data (routers): ZAC's router vs NEAT's, with and without waypoints,
# on the same placements (NEAT's placer on NEAT's generated array), every
# two-qubit stage of each of the 110 circuits, 8 circuits in parallel.
#   experiments/router/data/waypoints     NEAT and ZAC
#   experiments/router/data/no_waypoints  NEAT without waypoints, and ZAC again
#                                         (same placements, so ZAC must match)
# run_router_compare.py skips a circuit whose output exists, so a rerun resumes.
cd "$(dirname "$0")/../.."
J=$(mktemp)
python3 -c "from benchmark.suites import circuits;[print('qasmbench', b.split('/')[-1][:-5]) for b in circuits('qasmbench')]" > $J
for s in scalability/graphs scalability/random scalability/adder scalability/qft; do
  python3 -c "from benchmark.suites import circuits;[print('$s', b.split('/')[-1][:-5]) for b in circuits('$s')]" >> $J
done
xargs -P 8 -L 1 zsh -c '
  RC_ROUTERS=zac python3 experiments/router/run_router_compare.py "$0" "$1" > /dev/null 2>&1; echo "neat $0 $1 exit $?"
  RC_ROUTERS=zac RC_ABLATE_WAYPOINTS=1 python3 experiments/router/run_router_compare.py "$0" "$1" > /dev/null 2>&1; echo "nowp $0 $1 exit $?"' < $J
rm -f $J
rm -rf results/code    # run_router_compare.py can leave this behind
echo ALLDONE
