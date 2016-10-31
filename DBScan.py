import sys
import math

class DBSCAN(object):

    def __init__(self, eps=0, min_points=2):
        self.eps = eps
        self.min_points = min_points
        self.visited = []
        self.noise = []
        self.clusters = []
        self.dp = []

    def cluster(self, data_points):
        self.visited = []
        self.dp = data_points
        c = 0
        for point in data_points:
            if point not in self.visited:
                self.visited.append(point)
                neighbours = self.region_query(point)
                if len(neighbours) < self.min_points:
                    self.noise.append(point)
                else:
                    c += 1
                    self.expand_cluster(c, neighbours)

    def expand_cluster(self, cluster_number, p_neighbours):
        cluster = ("Cluster: %d" % cluster_number, [])
        self.clusters.append(cluster)
        new_points = p_neighbours
        while new_points:
            new_points = self.pool(cluster, new_points)

    def region_query(self, p):
        result = []
        for d in self.dp:
            distance = (((d[0] - p[0])**2 + (d[1] - p[1])**2 + (d[2] - p[2])**2)**0.5)
            if distance <= self.eps:
                result.append(d)
        return result

    def pool(self, cluster, p_neighbours):
        new_neighbours = []
        for n in p_neighbours:
            if n not in self.visited:
                self.visited.append(n)
                n_neighbours = self.region_query(n)
                if len(n_neighbours) >= self.min_points:
                    new_neighbours = self.unexplored(p_neighbours, n_neighbours)
            for c in self.clusters:
                if n not in c[1] and n not in cluster[1]:
                    cluster[1].append(n)
        return new_neighbours

    @staticmethod
    def unexplored(x, y):
        z = []
        for p in y:
            if p not in x:
                z.append(p)
        return z

    @staticmethod
    def array_2D_to_3D(data):
        x_i = 0
        y_i = 0
        output =[]
        for x in data:
            x_i += 1
            y_i = 0
            for y in x:
                y_i += 1
                output.append([x_i,y_i,y])
        return output


def main():
    data_0 = [
        [516, 517, 517, 518, 518, 516, 514, 515, 516, 515, 514, 515, 516, 516, 515, 515],
        [514, 515, 516, 518, 519, 517, 515, 516, 517, 516, 515, 516, 517, 516, 516, 515],
        [512, 514, 515, 518, 519, 518, 517, 517, 517, 517, 517, 517, 517, 517, 517, 517],
        [516, 516, 517, 518, 519, 518, 518, 518, 518, 518, 518, 519, 519, 518, 518, 518],
        [523, 522, 521, 521, 520, 520, 520, 520, 520, 520, 520, 521, 521, 520, 520, 519],
        [539, 538, 537, 535, 533, 531, 530, 529, 527, 527, 527, 527, 526, 525, 524, 523],
        [550, 548, 547, 544, 541, 539, 536, 534, 531, 530, 529, 529, 529, 527, 526, 525],
        [541, 540, 539, 536, 533, 531, 530, 528, 526, 525, 524, 521, 518, 517, 517, 517],
        [515, 515, 515, 512, 509, 508, 508, 508, 507, 505, 504, 505, 506, 506, 505, 506],
        [484, 487, 490, 489, 487, 487, 487, 487, 487, 489, 491, 489, 488, 491, 494, 495],
        [477, 480, 482, 482, 483, 483, 483, 482, 482, 482, 483, 485, 487, 487, 486, 487],
        [490, 491, 492, 492, 492, 492, 492, 492, 493, 493, 493, 494, 495, 495, 495, 495],
        [509, 509, 508, 508, 508, 508, 507, 508, 508, 508, 508, 508, 508, 507, 507, 506],
        [519, 519, 519, 519, 519, 519, 519, 519, 518, 518, 517, 516, 516, 515, 514, 513],
        [525, 525, 525, 526, 526, 527, 527, 525, 524, 523, 522, 521, 520, 519, 519, 518],
        [522, 522, 521, 521, 521, 521, 521, 519, 516, 516, 517, 517, 517, 517, 517, 517]]

    data = [[206, 214, 222, 226, 227, 221, 215, 214, 214, 210, 207, 211, 217, 219, 219, 216],
    [191, 201, 213, 222, 226, 217, 206, 202, 204, 211, 222, 220, 208, 206, 211, 213],
    [177, 188, 204, 214, 222, 223, 221, 212, 207, 216, 234, 239, 221, 208, 207, 214],
    [178, 191, 210, 225, 232, 237, 244, 237, 220, 212, 223, 234, 228, 220, 222, 229],
    [185, 200, 220, 231, 229, 238, 254, 255, 233, 213, 213, 222, 224, 227, 238, 244],
    [191, 211, 224, 220, 215, 232, 248, 242, 229, 224, 223, 221, 219, 225, 242, 251],
    [200, 222, 227, 216, 215, 232, 237, 224, 223, 240, 245, 235, 225, 227, 241, 245],
    [217, 240, 245, 232, 229, 237, 240, 229, 228, 243, 250, 241, 231, 232, 238, 233],
    [234, 247, 248, 242, 244, 244, 243, 239, 235, 237, 239, 239, 237, 237, 234, 227],
    [234, 239, 238, 231, 237, 240, 243, 245, 237, 232, 228, 231, 233, 232, 232, 234],
    [232, 230, 227, 221, 228, 236, 243, 247, 237, 233, 230, 229, 230, 229, 231, 235],
    [240, 245, 241, 233, 231, 238, 243, 244, 238, 238, 238, 238, 243, 241, 229, 221],
    [246, 259, 255, 252, 247, 245, 242, 239, 238, 235, 235, 242, 256, 253, 228, 209],
    [248, 263, 266, 262, 248, 236, 234, 237, 234, 227, 234, 252, 267, 259, 235, 221],
    [239, 249, 256, 254, 239, 223, 226, 237, 231, 222, 233, 252, 258, 252, 242, 239],
    [233, 240, 247, 246, 231, 221, 227, 232, 230, 228, 234, 242, 238, 232, 238, 246]]

    print data

    data1 = DBSCAN.array_2D_to_3D(data)
    print data1
    N = len(data1)
    if N == 0:
        return 0.0
    datasum = 0
    sum_mean_power =0
    for data_i in data:
        datasum += sum(data_i)

    data_mean = datasum / float(N)
    for data_i in data:
        sum_mean_power += sum((x - data_mean) ** 2 for x in data_i)
    ret = math.sqrt(sum_mean_power / (N - 1))

    print "SD is {0}".format(ret)
    dbfunc = DBSCAN(ret*1.414, 2)
    dbfunc.cluster(data1)
    print dbfunc.clusters
    print "partition in to {0} clusters".format(len(dbfunc.clusters))

if __name__ == '__main__':
    main()
