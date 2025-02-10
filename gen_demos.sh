#!/bin/bash

# folders="11A 12 12A 13 13A 14 14A"
# folders="12A 13 13A 14 14A"
#folders="11snake 12snake 14snake"
# folders='13snake'
folders="A20 A21 A22 A23 20snake 21snake 22snake 23snake"
for item in $folders; do
  echo "Generating from : $item"
  python lerobot/scripts/evalv2.py     -p ./local/models/$item     eval.n_episodes=100     eval.batch_size=100
done

