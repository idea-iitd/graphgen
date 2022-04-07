#include <bits/stdc++.h>
using namespace std;

auto start_time_for_loop = chrono::steady_clock::now();

const int inf = 10000000;
float time_limit;
// represents a general graph edge
struct Edge {
    int src;
    int dest;
    string lbl;
    Edge(int src, int dest, string lbl) {
        this->src = src;
        this->dest = dest;
        this->lbl = lbl;
    }
    Edge(int src, int dest) {
        this->src = src;
        this->dest = dest;
    }
};

// represents a min DFS style representation of an edge i,e a
// 5-tuple (i,j,labelA,EdgeLabel,labelB)
struct DfsEdge {
    int vertex1;
    int vertex2;
    string label1;
    string edge_label;
    string label2;

    DfsEdge(int vertex1, int vertex2, string label1, string edge_label,
            string label2) {
        this->vertex1 = vertex1;
        this->vertex2 = vertex2;
        this->label1 = label1;
        this->edge_label = edge_label;
        this->label2 = label2;
    }
    void print()  // prints a DFS style edge of 5 tuple
    {
        cout << "<" << vertex1 << "," << vertex2 << "," << label1 << ","
             << edge_label << "," << label2 << ">";
    }

    // Compare only edge and vertex labels
    bool compare_labels_less(DfsEdge e) {
        if (this->label1 < e.label1) {
            return true;
        }

        if (this->label1 == e.label1 && this->edge_label < e.edge_label) {
            return true;
        }
            
        if (this->label1 == e.label1 &&
            this->edge_label == e.edge_label && this->label2 < e.label2) {
            return true;
        }

        return false;
    }

    // operator overloading for comparison
    // Don't assume vertex1 is same and write complete comparator for code reuse and avoid confusion
    // DFS codes order - https://sites.cs.ucsb.edu/~xyan/papers/gSpan.pdf (Section 4)
    bool operator<(DfsEdge e) {
        // Both forward edges
        if (this->vertex2 > this->vertex1 && e.vertex2 > e.vertex1) {
            if (this->vertex2 < e.vertex2) {
                return true;
            } else if (this->vertex2 == e.vertex2) {
                if (compare_labels_less(e)) {
                    return true;
                }
            }
        }
        // Both backward edges
        else if (this->vertex2 < this->vertex1 && e.vertex2 < e.vertex1) {
            if (this->vertex1 < e.vertex1) {
                return true;
            } else if (this->vertex1 == e.vertex1 && this->vertex2 < e.vertex2) {
                return true;
            } else if (this->vertex1 == e.vertex1 && this->vertex2 == e.vertex2) {
                if (compare_labels_less(e)) {
                    return true;
                }
            }
        }
        // this is backward edge and e is forward edge
        else if (this->vertex2 < this->vertex1 && e.vertex2 > e.vertex1) {
            if (this->vertex1 < e.vertex2) {
                return true;
            }
        } 
        // this is forward edge and e is backward edge
        else if (this->vertex2 > this->vertex1 && e.vertex2 < e.vertex1) {
            if (this->vertex2 <= e.vertex1) {
                return true;
            }
        }

        return false;
    }

    // operator overloading to check equivalence
    bool operator==(DfsEdge e) {
        if (this->vertex1 != e.vertex1)
            return false;
        if (this->vertex2 != e.vertex2)
            return false;
        if (this->label1 != e.label1)
            return false;
        if (this->edge_label != e.edge_label)
            return false;
        if (this->label2 != e.label2)
            return false;
        return true;
    }
};  // class over of DFS style edges

// these variables kept only for these two comparators
vector<Edge> universal_edges;
vector<string> universal_labels;
int universal_u;
vector<int> universal_dfs_number;

// back normal edges are comparered
// differently than front normal edges
// acc to dfs rules
bool backEdgeComparator(int i, int j) {
    Edge e1 = universal_edges[i];
    Edge e2 = universal_edges[j];
    int v1 = (e1.src != universal_u) ? e1.src : e1.dest;
    int v2 = (e2.src != universal_u) ? e2.src : e2.dest;
    return (universal_dfs_number[v1] < universal_dfs_number[v2]);
}

// back normal edges are comparered
// differently than front normal edges
// acc to dfs rules
bool frontEdgeComparator(int i, int j) {
    Edge e1 = universal_edges[i];
    Edge e2 = universal_edges[j];
    if (e1.lbl != e2.lbl)
        return (e1.lbl < e2.lbl);
    int v1 = (e1.src != universal_u) ? e1.src : e1.dest;
    int v2 = (e2.src != universal_u) ? e2.src : e2.dest;
    return (universal_labels[v1] < universal_labels[v2]);
}

// prints full dfs code (not min!)
void print(vector<DfsEdge> code) {
    int i;
    for (i = 0; i < code.size(); ++i) {
        code[i].print();
        cout << endl;
    }
}

// graph class
struct Graph {
    int n, m;
    vector<string> labels;
    vector<vector<int>> adj_list;
    vector<bool> vertex_vis;
    vector<bool> edge_vis;
    vector<int> dfs_number;
    int disc;
    vector<Edge> edges;
    vector<DfsEdge> min_code;
    vector<pair<int, int>> min_vertex_code;
    bool min_code_set;
    vector<DfsEdge> code;
    vector<pair<int, int>> vertex_code;
    vector<int> parent;

    // graph constructor
    Graph(int _n) {
        this->n = _n;
        labels.clear();
        adj_list.resize(n);
        min_code_set = false;
    }
    bool isLesser(vector<DfsEdge> a, vector<DfsEdge> b) {
        int i;
        for (i = 0; i < min(a.size(), b.size()); ++i)
            if (a[i] < b[i])
                return true;
            else if (!(a[i] == b[i]))
                return false;
        return false;
    }
    bool isGreater(vector<DfsEdge> a, vector<DfsEdge> b) {
        int i;
        for (i = 0; i < min(a.size(), b.size()); ++i) {
            if (a[i] < b[i])
                return false;
            if (b[i] < a[i])
                return true;
        }
        return false;
    }

    void checkForMin(vector<DfsEdge> code) {
        if (!min_code_set || isLesser(code, min_code)) {
            min_code = code;
            min_vertex_code = vertex_code;
            min_code_set = true;
        }
    }

    // recurrsive function to explore dfs search from a given node u.
    int dfs(int u, int d) {
        //check if we have more time to calculate code
        auto lap = chrono::steady_clock::now();
        float time_till_now = chrono::duration_cast<chrono::milliseconds>(lap - start_time_for_loop).count();
        // printf("%f   %f\n",time_till_now,time_limit);
        if (time_limit > 0.0 && time_till_now > time_limit) {
            return 0;
        }

        // if current side has reacht = m then comapred it
        // with min code till and terminate this search
        if (code.size() == m) {
            checkForMin(code);
            return 1;
        }
        if (d == 0)  // number of further edges to be explored = 0
            return 0;

        vector<int> nei = adj_list[u];
        vector<int> back, front;  // seperate for back edges and front edges

        // Populate back and front edge lists
        for (int i = 0; i < nei.size(); ++i) {
            int v = (edges[nei[i]].src != u) ? edges[nei[i]].src : edges[nei[i]].dest;
            if (v == parent[u])
                continue;
            if (edge_vis[nei[i]])
                continue;
            if (vertex_vis[v] == true)
                back.push_back(nei[i]);
            else
                front.push_back(nei[i]);
        }

        int i;
        universal_u = u;
        universal_dfs_number = dfs_number;
        sort(back.begin(), back.end(), backEdgeComparator);
        sort(front.begin(), front.end(), frontEdgeComparator);

        // front and back are normal sorted list of normal front and back edges

        // these lists mark indexes in frontal list which
        // represents a similar edge to similar node
        vector<int> start, end;
        i = 0;  //(obviously they will be adjacent in the sorted list)
        // Populate start and end
        while (i < front.size()) {
            start.push_back(i);
            while (i < front.size()) {
                if (i == front.size() - 1) {
                    end.push_back(i);
                    i++;
                    break;
                }
                Edge e1 = edges[front[i]];
                Edge e2 = edges[front[i + 1]];
                int v1 = (e1.src != u) ? e1.src : e1.dest;
                int v2 = (e2.src != u) ? e2.src : e2.dest;
                if (e1.lbl == e2.lbl && labels[v1] == labels[v2])
                    i++;
                else {
                    end.push_back(i);
                    i++;
                    break;
                }
            }
        }

        // instead of numbers dealing in iterators to ease out code readability
        vector<vector<int>::iterator> start_it, end_it;
        vector<int>::iterator it = front.begin();
        i = 0;
        // Populate start_it and end_it
        while (it != front.end()) {
            start_it.push_back(it);
            int j;
            for (j = 0; j < end[i] - start[i]; ++j)
                it++;
            end_it.push_back(it);
            it++;
            i++;
        }

        // Populate code with back edges (Acc. to
        // dfs back edges get filled first)
        for (i = 0; i < back.size(); ++i) {
            if (edge_vis[back[i]])
                continue;
            edge_vis[back[i]] = true;
            int v =
                (edges[back[i]].src != u) ? edges[back[i]].src : edges[back[i]].dest;
            DfsEdge e(dfs_number[u], dfs_number[v], labels[u], edges[back[i]].lbl,
                      labels[v]);
            code.push_back(e);
            d--;
            vertex_code.push_back(make_pair(u, v));

            // these 3 checks whenever a new dfs edge is added to temporary code
            // if max size of code reached
            // if no further exploration required
            if (code.size() == m)
                checkForMin(code);
            if (d == 0)
                return 0;
            // if code explored till now
            // is already greater than min code dont explore further this path.
            if (min_code_set && isGreater(code, min_code)) {
                return 0;
            }
        }

        // store graph variables as from here paths of explore can be many
        vector<bool> vertex_vis_status = vertex_vis;
        vector<bool> edge_vis_status = edge_vis;
        vector<DfsEdge> code_status = code;
        vector<pair<int, int>> vertex_code_status = vertex_code;
        vector<int> dfs_number_status = dfs_number;
        vector<int> parent_status = parent;
        int disc_status = disc;

        for (i = 0; i < start.size(); ++i) {
            bool dec_i = false;
            // when we have various similar candidates
            // for further exloration for eg. single bond C
            if (end[i] - start[i] != 0) {
                vertex_vis_status = vertex_vis;
                edge_vis_status = edge_vis;
                code_status = code;
                vertex_code_status = vertex_code;
                dfs_number_status = dfs_number;
                parent_status = parent;
                disc_status = disc;

                int previous_code_size = code.size();
                // D represents depth till where we
                // go for comparing various options, we do DFID

                // This loop is just from comparing various option and knowing best,
                // this loop will not add to dfs code. it will reassign graph
                // variables at the end of loop again.
                for (int D = 1; D <= 2 * d; D *= 2) {
                    vector<vector<DfsEdge>> codes;
                    vector<int>::iterator it;
                    vector<int>::iterator end_plus = end_it[i];
                    end_plus++;
                    vector<DfsEdge> extended_code;
                    vector<int> rets;
                    int size = 0;
                    for (it = start_it[i]; it != end_plus; ++it) {
                        codes.push_back(code);
                        rets.push_back(0);
                    }
                    int num = 0;
                    for (it = start_it[i]; it != end_plus; ++it, ++num) {
                        int D_left = D;
                        int d_left = d;
                        vertex_vis = vertex_vis_status;
                        edge_vis = edge_vis_status;
                        code = code_status;
                        vertex_code = vertex_code_status;
                        dfs_number = dfs_number_status;
                        parent = parent_status;
                        disc = disc_status;

                        if (edge_vis[*it]) {
                            continue;
                        }
                        edge_vis[*it] = true;
                        int v = (edges[*it].src != u) ? edges[*it].src : edges[*it].dest;
                        if (!vertex_vis[v]) {
                            vertex_vis[v] = true;
                            dfs_number[v] = disc++;
                            DfsEdge e(dfs_number[u], dfs_number[v], labels[u], edges[*it].lbl,
                                      labels[v]);
                            code.push_back(e);
                            D_left--;
                            d_left--;
                            vertex_code.push_back(make_pair(u, v));
                            parent[v] = u;
                            int code_size = code.size();
                            int ret = dfs(v, min(D_left, d_left));
                            rets[num] = ret;
                            int extended_code_size = code.size() - code_size;
                            D_left -= extended_code_size;
                            d_left -= extended_code_size;
                        }
                        codes[num] = code;
                    }
                    int j;
                    if (codes.size() == 0)
                        break;
                    vector<DfsEdge> local_min_code = codes[0];
                    int perm = 0;
                    bool solo_min = true;

                    for (j = 1; j < codes.size(); ++j) {
                        if (isLesser(codes[j], local_min_code)) {
                            local_min_code = codes[j];
                            perm = j;
                            solo_min = true;
                        } else if (!isGreater(codes[j], local_min_code))
                            solo_min = false;
                    }

                    bool break_flag = false;
                    for (j = 0; j < codes.size(); ++j) {
                        if (!isLesser(codes[j], local_min_code) &&
                            !isGreater(codes[j], local_min_code)) {
                            if (rets[j] == 1) {
                                break_flag = true;
                                perm = j;
                            }
                        }
                    }

                    previous_code_size = code.size();

                    // fixing what changes has been bought by this D loop
                    vertex_vis = vertex_vis_status;
                    edge_vis = edge_vis_status;
                    code = code_status;
                    vertex_code = vertex_code_status;
                    dfs_number = dfs_number_status;
                    parent = parent_status;
                    disc = disc_status;

                    // if we found solo min then that's min of all
                    // options, or if it's last D iteration, we'll have
                    // to pick whatever min we have right now as one
                    // or if there is some non solo min path which completes all the
                    // edges, then that's it, we need stop looking further for best option
                    if (solo_min || D * 2 > 2 * d || break_flag) {
                        vector<int>::iterator it = start_it[i];
                        int temp_perm = perm;
                        while (temp_perm--)
                            it++;
                        int temp = *start_it[i];
                        *start_it[i] = *it;
                        *it = temp;

                        // remove selected one and restart process (D loop) on remaining
                        // options.
                        dec_i = true;
                        break;
                    }
                    if (break_flag) {
                        break;
                    }
                }
            }

            // from here till end of this function is taking dfs code at position
            // start_it[i]
            vector<int>::iterator it;
            vector<int>::iterator end = end_it[i];
            end++;
            for (it = start_it[i]; it == start_it[i]; ++it) {
                if (edge_vis[*it])
                    continue;
                edge_vis[*it] = true;
                int v = (edges[*it].src != u) ? edges[*it].src : edges[*it].dest;
                if (!vertex_vis[v]) {
                    vertex_vis[v] = true;
                    dfs_number[v] = disc++;
                    DfsEdge e(dfs_number[u], dfs_number[v], labels[u], edges[*it].lbl,
                              labels[v]);
                    code.push_back(e);
                    d--;
                    vertex_code.push_back(make_pair(u, v));
                    parent[v] = u;

                    if (code.size() == m)
                        checkForMin(code);
                    if (d == 0)
                        return 0;
                    if (min_code_set && isGreater(code, min_code)) {
                        return 0;
                    }

                    int code_size = code.size();
                    dfs(v, d);
                    int extended_code_size = code.size() - code_size;
                    d -= extended_code_size;
                    if (code.size() == m)
                        checkForMin(code);
                    if (d == 0)
                        return 0;
                } else {
                    DfsEdge e(dfs_number[u], dfs_number[v], labels[u], edges[*it].lbl,
                              labels[v]);
                    code.push_back(e);
                    vertex_code.push_back(make_pair(u, v));
                    d--;
                    if (code.size() == m)
                        checkForMin(code);
                    if (d == 0)
                        return 0;
                }
            }

            // if dec_i enabled that means we have taken one from similar
            // types and now we need to compare rest.
            if (dec_i) {
                start[i]++;
                start_it[i]++;
                i--;
            }
        }
        return 1;
    }
    int normal_dfs(int u, int d) {
        // printf("1\n");
        // if current side has reacht = m then comapred it
        // with min code till and terminate this search
        if (code.size() == m) {
            // printf("2\n");
            checkForMin(code);
            return 1;
        }
        if (d == 0)  // number of further edges to be explored = 0
            return 0;

        vector<int> nei = adj_list[u];
        vector<int> back, front;  // seperate for back edges and front edges

        // Populate back and front edge lists
        for (int i = 0; i < nei.size(); ++i) {
            int v = (edges[nei[i]].src != u) ? edges[nei[i]].src : edges[nei[i]].dest;
            if (v == parent[u])
                continue;
            if (edge_vis[nei[i]])
                continue;
            if (vertex_vis[v] == true)
                back.push_back(nei[i]);
            else
                front.push_back(nei[i]);
        }

        int i;
        universal_u = u;
        universal_dfs_number = dfs_number;
        sort(back.begin(), back.end(), backEdgeComparator);
        sort(front.begin(), front.end(), frontEdgeComparator);

        // front and back are normal sorted list of normal front and back edges

        // these lists mark indexes in frontal list which
        // represents a similar edge to similar node
        vector<int> start, end;
        i = 0;  //(obviously they will be adjacent in the sorted list)
        // Populate start and end
        while (i < front.size()) {
            start.push_back(i);
            end.push_back(i);
            i++;
            // while (i < front.size()) {
            //   if (i == front.size() - 1) {
            //     end.push_back(i);
            //     i++;
            //     break;
            //   }
            //   Edge e1 = edges[front[i]];
            //   Edge e2 = edges[front[i + 1]];
            //   int v1 = (e1.src != u) ? e1.src : e1.dest;
            //   int v2 = (e2.src != u) ? e2.src : e2.dest;
            //   if (e1.lbl == e2.lbl && labels[v1] == labels[v2])
            //     i++;
            //   else {
            //     end.push_back(i);
            //     i++;
            //     break;
            //   }
            // }
        }

        // instead of numbers dealing in iterators to ease out code readability
        vector<vector<int>::iterator> start_it, end_it;
        vector<int>::iterator it = front.begin();
        i = 0;
        // Populate start_it and end_it
        while (it != front.end()) {
            start_it.push_back(it);
            int j;
            for (j = 0; j < end[i] - start[i]; ++j)
                it++;
            end_it.push_back(it);
            it++;
            i++;
        }

        // Populate code with back edges (Acc. to
        // dfs back edges get filled first)
        for (i = 0; i < back.size(); ++i) {
            // printf("3\n");
            if (edge_vis[back[i]])
                continue;
            edge_vis[back[i]] = true;
            int v =
                (edges[back[i]].src != u) ? edges[back[i]].src : edges[back[i]].dest;
            DfsEdge e(dfs_number[u], dfs_number[v], labels[u], edges[back[i]].lbl,
                      labels[v]);
            code.push_back(e);
            d--;
            vertex_code.push_back(make_pair(u, v));

            // these 3 checks whenever a new dfs edge is added to temporary code
            // if max size of code reached
            // if no further exploration required
            if (code.size() == m)
                checkForMin(code);
            if (d == 0)
                return 0;
            // if code explored till now
            // is already greater than min code dont explore further this path.
            if (min_code_set && isGreater(code, min_code)) {
                return 0;
            }
        }

        // store graph variables as from here paths of explore can be many
        vector<bool> vertex_vis_status = vertex_vis;
        vector<bool> edge_vis_status = edge_vis;
        vector<DfsEdge> code_status = code;
        vector<pair<int, int>> vertex_code_status = vertex_code;
        vector<int> dfs_number_status = dfs_number;
        vector<int> parent_status = parent;
        int disc_status = disc;

        // printf("adding forward edges\n");
        // printf("code size = %d and d = %d\n",code.size(),d);

        for (i = 0; i < start.size(); ++i) {
            // from here till end of this function is taking dfs code at position
            // start_it[i]
            vector<int>::iterator it;
            vector<int>::iterator end = end_it[i];
            end++;
            for (it = start_it[i]; it == start_it[i]; ++it) {
                if (edge_vis[*it])
                    continue;
                edge_vis[*it] = true;
                int v = (edges[*it].src != u) ? edges[*it].src : edges[*it].dest;
                if (!vertex_vis[v]) {
                    vertex_vis[v] = true;
                    dfs_number[v] = disc++;
                    DfsEdge e(dfs_number[u], dfs_number[v], labels[u], edges[*it].lbl,
                              labels[v]);
                    code.push_back(e);
                    d--;
                    vertex_code.push_back(make_pair(u, v));
                    parent[v] = u;

                    if (code.size() == m)
                        checkForMin(code);
                    if (d == 0)
                        return 0;
                    if (min_code_set && isGreater(code, min_code)) {
                        return 0;
                    }

                    int code_size = code.size();
                    normal_dfs(v, d);
                    int extended_code_size = code.size() - code_size;
                    d -= extended_code_size;
                    if (code.size() == m)
                        checkForMin(code);
                    if (d == 0)
                        return 0;
                } else {
                    DfsEdge e(dfs_number[u], dfs_number[v], labels[u], edges[*it].lbl,
                              labels[v]);
                    code.push_back(e);
                    vertex_code.push_back(make_pair(u, v));
                    d--;
                    if (code.size() == m)
                        checkForMin(code);
                    if (d == 0)
                        return 0;
                }
            }

            // if dec_i enabled that means we have taken one from similar
            // types and now we need to compare rest.
        }
        return 1;
    }
    // gets min dfs code for a graph
    void get_min_dfscode(const char *outfile) {
        int i;
        min_code_set = false;
        vector<string> sortted_labels = labels;
        sort(sortted_labels.begin(), sortted_labels.end());
        string least_label = sortted_labels[0];

        auto start_time_for_loop = chrono::steady_clock::now();
        for (int src = 0; src < n; ++src) {
            // min dfs code can start only from vertex of least lexographic label
            if (labels[src] != least_label)
                continue;

            vertex_vis.resize(n);  // resetting graph dfs variables
            edge_vis.resize(m);
            parent.resize(n);
            for (i = 0; i < n; ++i) {
                vertex_vis[i] = false;
                parent[i] = -1;
            }
            for (i = 0; i < m; ++i)
                edge_vis[i] = false;
            dfs_number.resize(n);
            code.clear();
            vertex_code.clear();
            disc = 0;
            vertex_vis[src] = true;
            dfs_number[src] = disc++;  // reseting complete
            // start_time_for_loop = chrono::steady_clock::now();
            dfs(src, m);  // start dfs making process from source node src given m
                          // m edges are yet to be explored.
        }
        auto lap = chrono::steady_clock::now();
        float time_till_now = chrono::duration_cast<chrono::milliseconds>(lap - start_time_for_loop).count();
        // printf("%f   %f\n",time_till_now,time_limit);

        if (!min_code_set) {
            cout << "XTLE: PRINTING NORMAL DFS ANSWER |V| = " << n << "  |E| = " << m << endl;
            for (int src = 0; src < n; ++src) {
                // min dfs code can start only from vertex of least lexographic label
                if (labels[src] != least_label)
                    continue;

                vertex_vis.resize(n);  // resetting graph dfs variables
                edge_vis.resize(m);
                parent.resize(n);
                for (i = 0; i < n; ++i) {
                    vertex_vis[i] = false;
                    parent[i] = -1;
                }
                for (i = 0; i < m; ++i)
                    edge_vis[i] = false;
                dfs_number.resize(n);
                code.clear();
                vertex_code.clear();
                disc = 0;
                vertex_vis[src] = true;
                dfs_number[src] = disc++;  // reseting complete
                start_time_for_loop = chrono::steady_clock::now();
                normal_dfs(src, m);  // start dfs making process from source node src given m
                                     // m edges are yet to be explored.
            }
        } else if (time_limit > 0.0 && time_till_now > time_limit) {
            cout << "TLE: PRINTING SUBOPTIMAL ANSWER |V| = " << n << "  |E| = " << m << endl;
        }

        ofstream ofile;
        ofile.open(outfile, ios::out);

        for (i = 0; i < min_code.size(); ++i)  // print min dfs code
        {
            ofile << " < " << min_code[i].vertex1 << " , " << min_code[i].vertex2
                  << " , " << min_code[i].label1 << " , " << min_code[i].edge_label
                  << " , " << min_code[i].label2 << " > ";
            ofile << endl;
        }

        ofile.close();
    }
};

int main(int argc, char *argv[]) {
    const char *outfile = argv[1];
    const float time_per_edge = atof(argv[2]);
    int start = clock();
    int n;
    cin >> n;
    Graph g(n);

    for (int i = 0; i < g.n; ++i) {
        string lbl;
        cin >> lbl;
        g.labels.push_back(lbl);
    }

    universal_labels = g.labels;

    cin >> g.m;
    for (int i = 0; i < g.m; ++i) {
        int src, dest;
        string lbl;
        cin >> src >> dest >> lbl;
        Edge e(src, dest, lbl);
        g.edges.push_back(e);
        g.adj_list[src].push_back(g.edges.size() - 1);
        g.adj_list[dest].push_back(g.edges.size() - 1);
    }

    universal_edges = g.edges;

    for (int i = 0; i < g.n; ++i) {
        sort(g.adj_list[i].begin(), g.adj_list[i].end());
    }

    time_limit = time_per_edge * g.m * 1000;
    g.get_min_dfscode(outfile);
}
