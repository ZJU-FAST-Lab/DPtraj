roslaunch random_map_generator test.launch            & sleep 1; #not sending map now
roslaunch plan_manage plan_node.launch & sleep 1;
wait

