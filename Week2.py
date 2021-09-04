"""
You can create any other helper funtions.
Do not modify the given functions
"""


import queue


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
    priority = queue.PriorityQueue()
    priority.put((heuristic[start_point], ([start_point], start_point, 0)))

    while(priority.qsize()):
        cost2, nodes = priority.get()
        path = nodes[0]
        curr = nodes[1]
        nodeCost = nodes[2]

    n = len(cost)
    visited = [0 for i in range(n)]

    if visited[curr] == 0:
        visited[curr] = 1
    if curr in goals:
        return path

    for next in range(1, n):
        if cost[curr][next] > 0 and visited[next] == 0:
            total = nodeCost + cost[curr][next]
            cost2 = total + heuristic[next]
            path.append(next)
            priority.put((cost2, (path, next, total)))

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

    stack = queue.LifoQueue(maxsize = 0)

    n = len(cost)
    visited = [0 for i in range(n)]
    parents = {}
    stack.put((start_point, [start_point]))

    while(stack.qsize()):
        node, path = stack.get()

        if visited[node] == 0:
            visited[node] = 1
        else:
            continue
        if node in goals:
            path = [node]
            if start_point == node:
                return path
            cur = node
            while not cur == start_point:
                cur = parents[cur]
                path.insert(0, cur)
            return path

        else:
            for i in range(len(cost[node]) - 1, 0, -1):
                if visited[i] or cost[node][i] in [0, -1]:
                    continue
                parents[i] = node
                stack.put((i, path))

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
