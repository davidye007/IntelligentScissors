import math, pygame, scipy, heapq, timeit, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

def xy_to_vertex(x, y, w):
    return x + y*w

def vertex_to_xy(v, w):
    return (v%w, v//w)

# Takes in a pygame image and returns the appropriate graph.
# A graph is represented as a Python dictionary whose keys are
# vertices v and entries are lists of (vertex, weight) pairs.
def image_to_graph(image, width, height):
    # Get R,G,B arrays from image, correct array orientation by flipping x and y axis
    rgb_values = pygame.surfarray.array3d(image)
    rgb_values =  np.swapaxes(rgb_values, 0, 1)
    # Calculate luma array
    luma = 0.299*rgb_values[:,:,0] + 0.587*rgb_values[:,:,1] + 0.114*rgb_values[:,:,2]
    # Initialize convolution kernels
    g_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    g_y = np.array([[-1,2,-1],[0,0,0],[1,2,1]])
    delta = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # Laplacian matrix
    Lap = scipy.ndimage.correlate(luma,delta,mode = 'constant', cval = 0.0)
    # Gradient in x direction
    G_x = scipy.ndimage.correlate(luma,g_x,mode = 'constant', cval = 0.0)
    # Gradient in y direction
    G_y = scipy.ndimage.correlate(luma,g_y,mode = 'constant', cval = 0.0)
    # Magnitude of gradient
    G = np.sqrt(np.square(G_x)+np.square(G_y))
    # Normalize gradient
    m = np.amax(G)
    G_normalized = np.divide(G,m)
    start = timeit.default_timer()
    # Calculate zero-crossing
    Z = np.ones_like(Lap).astype(np.float16)
    for y in range(height):
        for x in range(width):
            neighbors = [(-1, 0), (1, 0), (0, 1), (0, -1)]
            curr_sign = np.sign(Lap[y][x])
            curr_abs = abs(Lap[y][x])
            for (dx, dy) in neighbors:
                if 0 <= x+dx < width and 0 <= y+dy < height:
                    if((curr_abs < abs(Lap[y+dy][x+dx])) and curr_sign!=np.sign(Lap[y+dy][x+dx])):
                        Z[y][x] = 0
    im = plt.imshow(Z, cmap='Greys')
    plt.show()
    # Array of costs of every pixel
    C = 0.5*Z + 0.5*(1-G_normalized)
    im = plt.imshow(C, cmap='Greys')
    plt.show()
    # Horizontal edge weights
    Crop_1 = np.delete(C, -1, axis = 1) # delete right most column
    Crop_2 = np.delete(C, 0, axis = 1) # delete left most column
    Horizontal_Weights = np.divide(Crop_1+Crop_2, 2)
    Crop_3 = np.delete(C, -1, axis = 0) # delete bottom most row
    Crop_4 = np.delete(C, 0, axis = 0) # delete bottom most row
    Vertical_Weights = np.divide(Crop_3+Crop_4, 2)


    # return graph representation of image optimized
    ret = {v: {} for v in range(width*height)}
    for y in range(height):
        for x in range(width):
            v = xy_to_vertex(x, y, width)
            if 0 <= x+1 < width:
                w = xy_to_vertex(x+1, y, width)
                ret[v][w] = Horizontal_Weights[y,x]
                ret[w][v] = Horizontal_Weights[y,x]
            if 0 <= y+1 < height:
                w = xy_to_vertex(x, y+1, width)
                ret[v][w] = Vertical_Weights[y,x]
                ret[w][v] = Vertical_Weights[y,x]
    return ret

# Returns the shortest path from the source node to target node and original image.
def shortest_path_optimized(graph, source, target):
    ret = {source: (0, None)}
    # create heap
    h = []
    for v in graph[source]:
        heapq.heappush(h, [graph[source][v], source, v]) # weight, prev, curr
    while target not in ret:
        min_cost_list = heapq.heappop(h)
        minimum_cost = min_cost_list[0]
        min_cost_vertex = min_cost_list[-1]
        while min_cost_vertex in ret:
            min_cost_list = heapq.heappop(h)
            min_cost_vertex = min_cost_list[-1]
        ret[min_cost_vertex] = (min_cost_list[0], min_cost_list[1]) # ret[curr] = (weight, prev)

        for w in graph[min_cost_vertex]:
            if w in ret:
                continue
            new_cost = minimum_cost + graph[min_cost_vertex][w]
            heapq.heappush(h, [new_cost, min_cost_vertex, w])

    path = [target]
    while path[-1] != source:
        path.append(ret[path[-1]][1])

    return path[::-1]

# Returns the shortest path from the source node to target node and new image that
# visualizes all vertices visited.
def shortest_path_optimized_vis(graph, source, target, image):
    image_drawn = image.copy()
    image_rect = image.get_rect()
    width = image_rect.width
    ret = {source: (0, None)}
    # create heap
    h = []
    for v in graph[source]:
        heapq.heappush(h, [graph[source][v], source, v]) # weight, prev, curr
        pixel_color = image_drawn.get_at(vertex_to_xy(v, width))
        pixel_color[0] = int(pixel_color[0]*0.8)
        pixel_color[1] = int(pixel_color[1]*0.8)
        pixel_color[2] = int(pixel_color[2]*0.8)
        image_drawn.set_at(vertex_to_xy(v, width), (pixel_color))
        # image_drawn.set_at(vertex_to_xy(v, width), (255,0,0))
    while target not in ret:
        # shows only those pixels currently in heap
        # image_drawn = image.copy()
        # image_rect = image.get_rect()
        # width = image_rect.width
        # for i in h:
        #     v = i[2]
        #     # pixel_color = image_drawn.get_at(vertex_to_xy(v, width))
        #     # pixel_color[0] = int(pixel_color[2]*0.75)
        #     # pixel_color[1] = int(pixel_color[2]*0.75)
        #     # pixel_color[2] = int(pixel_color[2]*0.75)
        #     # image_drawn.set_at(vertex_to_xy(v, width), pixel_color)
        #     image_drawn.set_at(vertex_to_xy(v, width), (255,0,0))
        min_cost_list = heapq.heappop(h)
        minimum_cost = min_cost_list[0]
        min_cost_vertex = min_cost_list[-1]
        while min_cost_vertex in ret:
            min_cost_list = heapq.heappop(h)
            min_cost_vertex = min_cost_list[-1]
        ret[min_cost_vertex] = (min_cost_list[0], min_cost_list[1]) # ret[curr] = (weight, prev)

        for w in graph[min_cost_vertex]:
            if w in ret:
                continue
            new_cost = minimum_cost + graph[min_cost_vertex][w]
            heapq.heappush(h, [new_cost, min_cost_vertex, w])
            # shades in pixels visited to be darker
            pixel_color = image_drawn.get_at(vertex_to_xy(w, width))
            pixel_color[0] = int(pixel_color[0]*0.8)
            pixel_color[1] = int(pixel_color[1]*0.8)
            pixel_color[2] = int(pixel_color[2]*0.8)
            image_drawn.set_at(vertex_to_xy(w, width), pixel_color)


    path = [target]
    while path[-1] != source:
        path.append(ret[path[-1]][1])

    return path[::-1],image_drawn

# Returns the shortest path from the source node to target node and original image
def shortest_path_naive(graph, source, target):
    ret = {source: (0, None)}
    todo = {v: (graph[source][v], source) for v in graph[source]}
    while target not in ret:
        minimum_cost = math.inf
        minimum_cost_vertex = None
        # TODO: This for-loop needs to be sped up by using a heap / priority-queue.
        for v in todo:
            cost, _ = todo[v]
            # highlights all pixals in todo in red (search for minimum weight in these pixels)
            if cost < minimum_cost:
                minimum_cost = cost
                minimum_cost_vertex = v
        ret[minimum_cost_vertex] = todo.pop(minimum_cost_vertex)
        for w in graph[minimum_cost_vertex]:
            if w in ret:
                continue
            new_cost = minimum_cost + graph[minimum_cost_vertex][w]
            if w not in todo or todo[w][0] > new_cost:
                todo[w] = (new_cost, minimum_cost_vertex)

    path = [target]
    while path[-1] != source:
        path.append(ret[path[-1]][1])

    return path[::-1]


# Returns the shortest path from the source node to target node and new image
# visualizing all verticies in todo (in which min vertex must be found).
def shortest_path_naive_vis(graph, source, target, image):
    # TODO: Implement an efficient variant of Dijkstra's algorithm to find the shortest path from source to target.
    # A naive implementation is provided for you below.
    image_drawn = image.copy()
    image_rect = image.get_rect()
    width = image_rect.width


    ret = {source: (0, None)}
    todo = {v: (graph[source][v], source) for v in graph[source]}
    while target not in ret:
        image_drawn = image.copy()
        image_rect = image.get_rect()
        width = image_rect.width
        minimum_cost = math.inf
        minimum_cost_vertex = None

        # TODO: This for-loop needs to be sped up by using a heap / priority-queue.
        for v in todo:
            cost, _ = todo[v]
            # highlights all pixals in todo in red (search for minimum weight in these pixels)
            image_drawn.set_at(vertex_to_xy(v, width), (255,0,0))
            if cost < minimum_cost:
                minimum_cost = cost
                minimum_cost_vertex = v
        ret[minimum_cost_vertex] = todo.pop(minimum_cost_vertex)
        for w in graph[minimum_cost_vertex]:
            if w in ret:
                continue
            new_cost = minimum_cost + graph[minimum_cost_vertex][w]
            if w not in todo or todo[w][0] > new_cost:
                todo[w] = (new_cost, minimum_cost_vertex)

    path = [target]
    while path[-1] != source:
        path.append(ret[path[-1]][1])

    return path[::-1],image_drawn

## for benchmarking speed of naive dijkstra's
def shortest_path_bench(graph, source):
    ret = {source: (0, None)}
    todo = {v: (graph[source][v], source) for v in graph[source]}
    while len(ret)<len(graph):
        minimum_cost = math.inf
        minimum_cost_vertex = None
        for v in todo:
            cost, _ = todo[v]
            if cost < minimum_cost:
                minimum_cost = cost
                minimum_cost_vertex = v
        ret[minimum_cost_vertex] = todo.pop(minimum_cost_vertex)
        for w in graph[minimum_cost_vertex]:
            if w in ret:
                continue
            new_cost = minimum_cost + graph[minimum_cost_vertex][w]
            if w not in todo or todo[w][0] > new_cost:
                todo[w] = (new_cost, minimum_cost_vertex)
    return len(ret)

## for benchmarking speed of optimized dijkstra's
def shortest_path_optimized_bench(graph, source):
    ret = {source: (0, None)}
    # create heap
    h = []
    for v in graph[source]:
        heapq.heappush(h, [graph[source][v], source, v]) # weight, prev, curr
    while len(ret)<len(graph):
        min_cost_list = heapq.heappop(h)
        minimum_cost = min_cost_list[0]
        min_cost_vertex = min_cost_list[-1]
        while min_cost_vertex in ret:
            min_cost_list = heapq.heappop(h)
            min_cost_vertex = min_cost_list[-1]
        ret[min_cost_vertex] = (min_cost_list[0], min_cost_list[1]) # ret[curr] = (weight, prev)

        for w in graph[min_cost_vertex]:
            if w in ret:
                continue
            new_cost = minimum_cost + graph[min_cost_vertex][w]
            heapq.heappush(h, [new_cost, min_cost_vertex, w])
    return len(ret)