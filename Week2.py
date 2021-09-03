def helperFunc(visited, cost, start_point, goals, path):

    path += [start_point]
    visited[start_point] = 1

    if start_point in goals:
        return path
    
    costList = enumerate(cost[start_point])

    for curr, length in costList:
        if visited[curr] or not curr or length <= 0:
            continue
        helperFunc(cost, curr, goals, visited, path)
        return path

    return path

"""
You can create any other helper funtions.
Do not modify the given functions
"""

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    
    path = []
    
    # TODO
    
    queue = [[start_point, 0]]

    visited = [[0, -1, None] for i in cost]
    g = [0 for i in cost]

    while len(queue) != 0:
        node, f = queue.pop(0)

        if node in goals:
            print(node)
            return

        for neighbour, dist in enumerate(cost[node]):
            if dist <= 0:
                continue

            g[neighbour] = g[node] + dist
            f = g[neighbour] + heuristic[neighbour]

            if not visited[neighbour][0] or visited[neighbour][1] > f:
                visited[neighbour] = [1, f, node]
                # insert into priority queue/min heap
                

        path = sorted(queue, key=lambda x: x[1])[0]

    # DONE
    
    return path


def DFS_Traversal(cost, start_point, goals):

    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    
    path = []
    
    # TODO
    
    visited = [0 for i in cost]
    path = helperFunc(visited, cost, start_point, goals, path)
    
    # DONE
    
    return path
    
    


'''
cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1],
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
start = 1
goals = [6, 7, 10]

print(DFS_Traversal(cost,start, goals))
'''
