# Find biconnected components in a given undirected graph

from collections import defaultdict
import group
import time

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
        self.Time = 0
        self.count = 0

    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def BCCUtil(self, u, parent, low, disc, st):
        children = 0

        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
        result = []

        for v in self.graph[u]:
            if disc[v] == -1:
                parent[v] = u
                children += 1
                st.append((u, v))
                result += self.BCCUtil(v, parent, low, disc, st)

                low[u] = min(low[u], low[v])

                if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
                    self.count += 1
                    w = -1

                    connectedGraph = set()
                    while w != (u, v):
                        w = st.pop()
                        connectedGraph.add(w[0])
                        connectedGraph.add(w[1])
                    result.append(group.Group(connectedGraph, time.time(), False))
                    
            elif v != parent[u] and low[u] > disc[v]:
                low[u] = min(low[u], disc[v])
                st.append((u, v))

        return result

    def BCC(self):
        disc = defaultdict(lambda: -1)
        low = defaultdict(lambda: -1)
        parent = defaultdict(lambda: -1)
        st = []
        result = []

        for i in self.graph.keys():
            if disc[i] == -1:
                result += self.BCCUtil(i, parent, low, disc, st)

                if st:
                    self.count = self.count + 1

                    connectedGraph = set()
                    while st:
                        w = st.pop()
                        connectedGraph.add(w[0])
                        connectedGraph.add(w[1])
                    result.append(group.Group(connectedGraph, time.time(), False))
        
        return result

if __name__ == "__main__":
    g = Graph(12) 
    g.addEdge(0, 1) 
    g.addEdge(1, 2) 
    g.addEdge(1, 3) 
    g.addEdge(2, 3) 
    g.addEdge(2, 4) 
    g.addEdge(3, 4) 
    g.addEdge(1, 5) 
    g.addEdge(0, 6) 
    g.addEdge(5, 6) 
    g.addEdge(5, 7) 
    g.addEdge(5, 8) 
    g.addEdge(7, 8) 
    g.addEdge(8, 9) 
    g.addEdge(10, 11) 
  
    print(g.BCC()); 
    print ("Above are % d biconnected components in graph" %(g.count)); 
