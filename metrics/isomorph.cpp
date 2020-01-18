#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/labeled_graph.hpp"
#include "boost/graph/vf2_sub_graph_iso.hpp"
#include "boost/property_map/property_map.hpp"

using namespace std;

// Boiler plate code
typedef boost::property<boost::edge_name_t, int> edge_prop;
typedef boost::property<boost::vertex_name_t, string, boost::property<boost::vertex_index_t, string> > vertex_prop;

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, vertex_prop, edge_prop> Graph;

typedef boost::property_map<Graph, boost::vertex_name_t>::type vertex_name_map_t;
typedef boost::property_map_equivalent<vertex_name_map_t, vertex_name_map_t> vertex_comp_t;

typedef boost::property_map<Graph, boost::edge_name_t>::type edge_name_map_t;
typedef boost::property_map_equivalent<edge_name_map_t, edge_name_map_t> edge_comp_t;

template <typename Graph1,
          typename Graph2>
class my_call_back {
   public:
    // constructor
    my_call_back(const Graph1& graph1, const Graph2& graph2) : graph1_(graph1), graph2_(graph2) {}

    template <typename CorrespondenceMap1To2, typename CorrespondenceMap2To1>

    bool operator()(CorrespondenceMap1To2 f, CorrespondenceMap2To1) {
        return true;
    }

   private:
    const Graph1& graph1_;
    const Graph2& graph2_;
};

// Boiler plate code
bool is_subgraph_isomorphic(Graph smallGraph, Graph bigGraph) {
    vertex_comp_t vertex_comp =
        boost::make_property_map_equivalent(boost::get(boost::vertex_name, smallGraph), boost::get(boost::vertex_name, bigGraph));
    edge_comp_t edge_comp =
        boost::make_property_map_equivalent(boost::get(boost::edge_name, smallGraph), boost::get(boost::edge_name, bigGraph));
    my_call_back<Graph, Graph> callback(smallGraph, bigGraph);
    bool res = boost::vf2_subgraph_mono(smallGraph, bigGraph, callback, boost::vertex_order_by_mult(smallGraph),
                                        boost::edges_equivalent(edge_comp).vertices_equivalent(vertex_comp));
    return res;
}

void read_input(string inputfile, vector<Graph>& G) {
    ifstream infile;
    infile.open(inputfile);
    string line;
    int id = -1;
    while (getline(infile, line)) {
        if (line.at(0) == '#') {
            continue;
        }

        if (line.at(0) == 't') {
            id++;
            G.resize(id + 1);
            continue;
        }

        stringstream stream(line);
        char c;
        stream >> c;

        if (c == 'v') {
            int nd;
            string lbl;
            stream >> nd;
            stream >> lbl;
            boost::add_vertex(vertex_prop(lbl), G[id]);
        } else if (c == 'u') {
            int nd1, nd2, lbl;
            stream >> nd1 >> nd2 >> lbl;
            boost::add_edge(nd1, nd2, edge_prop(lbl), G[id]);
        }
    }
    infile.close();
}

int main(int argc, char* argv[]) {
    vector<Graph> G, g;
    string graphfile = argv[1];
    string subgraphfile = argv[2];
    int large = atoi(argv[3]);

    read_input(graphfile, G);
    read_input(subgraphfile, g);

    omp_set_num_threads(48);

    if (large == 0) {
        vector<int> unique(g.size(), 1);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < g.size(); i++) {
            for (int j = 0; j < G.size(); j++) {
                if (is_subgraph_isomorphic(g[i], G[j])) {
                    unique[i] = 0;
                    break;
                }
            }

#pragma omp critical
            {
                cout << i << " " << unique[i] << endl;
            }
        }
    } else {
        vector<int> unique(G.size(), 1);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < G.size(); i++) {
            for (int j = 0; j < g.size(); j++) {
                if (is_subgraph_isomorphic(g[j], G[i])) {
                    unique[i] = 0;
                    break;
                }
            }

#pragma omp critical
            {
                cout << i << " " << unique[i] << endl;
            }
        }
    }

    return 0;
}