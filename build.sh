mkdir -p bin

g++ -std=c++11 dfscode/dfs_code.cpp -o bin/dfscode -O3
g++ -std=c++11 metrics/orca/orca.cpp -o bin/orca -O3
g++ -std=c++11 metrics/isomorph.cpp -O3 -o bin/subiso -fopenmp -I$HOME/boost/include
g++ -std=c++11 metrics/unique.cpp -O3 -o bin/unique -fopenmp -I$HOME/boost/include