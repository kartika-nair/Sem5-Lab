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

    stack.put((start_point, [start_point]))

    while(stack.qsize()):
        node, path = stack.get()
    if visited[node] == 0:
        visited[node] = 1
    if node in goals:
        return path
                
    else:
        for next in range(n-1, 0, -1):
            if cost[node][next] > 0:
                if visited[next] == 0:
                    path.append(next)
                    stack.put((next, path))
    
    # DONE
    
    return path
