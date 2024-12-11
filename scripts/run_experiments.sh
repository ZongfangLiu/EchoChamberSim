#network_types=("small_world" "scale_free" "random")
network_types=("scale_free")
# "similarity"
#network_types=("small_world")
network_types=("scale_free")
recommendations=("similarity")
topics=("euthanasia")

for network_type in "${network_types[@]}"; do
  for recommendation in "${recommendations[@]}"; do
    for topic in "${topics[@]}"; do
      python main.py --max_interactions -1 --recommendation "$recommendation" --topic "$topic" --network_type "$network_type" --num_agents 50 --gpt_temp 1 --seed 50
    done
  done
done
