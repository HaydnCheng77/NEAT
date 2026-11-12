#!/bin/zsh
# Table II data (placers and placement objectives): the five placers at 10
# evenly spaced two-qubit stages of each of the 110 circuits (the 10 QASMBench
# circuits and every size of GraphState, Random, Adder and QFT, n = 10..1000),
# 8 circuits in parallel. Output: experiments/placer/data/.
# run_five_placers.py skips a circuit whose output exists, so a rerun resumes.
# Progress: results/table2.progress (one line per stage).
cd "$(dirname "$0")/../.."
mkdir -p results
J=$(mktemp)
python3 -c "from benchmark.suites import circuits;[print('qasmbench', b.split('/')[-1][:-5]) for b in circuits('qasmbench')]" > $J
for s in scalability/qft scalability/adder scalability/random scalability/graphs; do
  python3 -c "from benchmark.suites import circuits;[print('$s', b.split('/')[-1][:-5]) for b in reversed(circuits('$s'))]" >> $J
done
xargs -P 8 -L 1 zsh -c 'FP_SKIP_WHOLE=1 FP_PROGRESS=results/table2.progress FP_STAGE_LIMIT=3600 python3 experiments/placer/run_five_placers.py "$0" "$1" 10 > /dev/null 2>> results/table2.err; echo "$(date +%H:%M:%S) $0 $1 exit $?"' < $J
rm -f $J
echo ALLDONE
