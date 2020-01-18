import os
import subprocess
import tempfile
import pickle
import networkx as nx


def get_min_dfscode(G, temp_path=tempfile.gettempdir()):
    input_fd, input_path = tempfile.mkstemp(dir=temp_path)

    with open(input_path, 'w') as f:
        vcount = len(G.nodes)
        f.write(str(vcount) + '\n')
        i = 0
        d = {}
        for x in G.nodes:
            d[x] = i
            i += 1
            f.write(str(G.nodes[x]['label']) + '\n')

        ecount = len(G.edges)
        f.write(str(ecount) + '\n')
        for (u, v) in G.edges:
            f.write(str(d[u]) + ' ' + str(d[v]) +
                    ' ' + str(G[u][v]['label']) + '\n')

    output_fd, output_path = tempfile.mkstemp(dir=temp_path)

    dfscode_bin_path = 'bin/dfscode'
    with open(input_path, 'r') as f:
        subprocess.call([dfscode_bin_path, output_path, '2'], stdin=f)

    with open(output_path, 'r') as dfsfile:
        dfs_sequence = []
        for row in dfsfile.readlines():
            splited_row = row.split()
            splited_row = [splited_row[2 * i + 1] for i in range(5)]
            dfs_sequence.append(splited_row)

    os.close(input_fd)
    os.close(output_fd)

    try:
        os.remove(input_path)
        os.remove(output_path)
    except OSError:
        pass

    return dfs_sequence


def graph_from_dfscode(dfscode):
    graph = nx.Graph()

    for dfscode_egde in dfscode:
        i, j, l1, e, l2 = dfscode_egde
        graph.add_node(int(i), label=l1)
        graph.add_node(int(j), label=l2)
        graph.add_edge(int(i), int(j), label=e)

    return graph


if __name__ == '__main__':
    with open(os.path.expanduser('~/MTP/data/dataset/ENZYMES/graphs/graph180.dat'), 'rb') as f:
        G = pickle.load(f)

    dfs_code = get_min_dfscode(G)
    print(len(dfs_code), G.number_of_edges())
    for code in dfs_code:
        print(code)
