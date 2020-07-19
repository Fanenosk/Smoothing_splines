import numpy as np
import matplotlib.pyplot as plt
import Spline as splpckg
import Point as pnt
import math


class PlotSpline:
    def __init__(self, spline):
        self.m_delta = 0
        self.p = 0
        self.m_error = 0
        self.m_penalty = self.penalty(spline)

    def penalty(self, spline):
        knots = spline.get_knots()
        g = spline.get_internal_knots_num()
        self.m_penalty = 0
        for i in range(g + 1):
            self.m_penalty += 1.0 / (knots[i + 1] - knots[i])
        self.m_error = self.m_delta + self.p * self.m_penalty
        return self.m_penalty

    def delta(self, spline, points, smoothing_weight):
        self.m_delta = 0
        for point in points:
            e = point.w * (point.y - spline.get_value(point.x))
            self.m_delta += e * e
        nu = 0
        if smoothing_weight > 0:
            k = spline.get_degree()
            g = spline.get_internal_knots_num()
            coefficients = spline.get_coefficients()
            for q in range(k + 1, g + k + 1):
                e = 0
                for i in range(q - k - 1, q + 1):
                    e += coefficients[i] * spline.get_lead_derivative_difference(i, q)
                    nu += e * e
            nu *= smoothing_weight
        self.m_delta += nu
        self.m_error = self.m_delta + self.p * self.m_penalty
        return self.m_delta

    @staticmethod
    def norm(v):
        return sum(i * i for i in v)

    @staticmethod
    def approximate(spline, points, smoothing_weight):
        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        n = len(points)
        coefficients = [0] * (g + k + 1)
        A = [[0 for i in range(g + k + 1)] for j in range(g + k + 1)]

        # spline error
        l = 0
        for r in range(n):
            l = spline.get_left_node_index(points[r].x, l)
            if l < 0:
                return
            b_splines = spline.b_splines(points[r].x, k)
            for i in range(k + 1):
                w_sq = points[r].w * points[r].w
                for j in range(i + 1):
                    A[i + l - k][j + l - k] += w_sq * b_splines[i] * b_splines[j]
                coefficients[i + l - k] += w_sq * points[r].y * b_splines[i]

        # smoothing error
        if smoothing_weight > 0:
            for q in range(g):
                for i in range(q, q + k + 2):
                    ai = spline.get_lead_derivative_difference(i, q + k + 1)
                    for j in range(q, i + 1):
                        A[i][j] += smoothing_weight * ai * spline.get_lead_derivative_difference(j, q + k + 1)

        for i in range(g + k + 1):
            for j in range(i):
                A[j][i] = A[i][j]

        # decompose A = LU
        L = np.linalg.cholesky(A)

        # solve Lx = r
        for i in range(g + k + 1):
            for j in range(i):
                coefficients[i] -= L[i][j] * coefficients[j]
            coefficients[i] /= L[i][i]

        # solve Uy = x
        for i in reversed(range(g + k + 1)):
            for j in reversed(range(i + 1, g + k + 1)):
                coefficients[i] -= L[j][i] * coefficients[j]
            coefficients[i] /= L[i][i]

        spline.set_coefficients(coefficients)

        return True

    @staticmethod
    def initiate_grid(spline, points):
        k = spline.get_degree()
        g = spline.get_internal_knots_num()
        knots = [0] * (g + 2)
        knots[0] = spline.get_left_bound()
        knots[g + 1] = spline.get_right_bound()
        n = len(points)

        unique_size = 0
        index = 0

        while index < n and points[index].x < knots[g + 1]:
            if index != 0 and points[index].x != points[index - 1].x:
                unique_size += 1
            index += 1

        # not enough data points
        if unique_size <= 0:
            return False

        # number of knots should be less than n - k for n points with unique x
        if unique_size < g + k + 1:
            return False

        points_per_knot = unique_size / (g + 1)
        knot_index = 1
        i = 1
        counter = 0

        while knot_index < g + 1:
            while counter < knot_index * points_per_knot or points[i].x == points[i - 1].x:
                if points[i].x != points[i - 1].x:
                    counter += 1
                i += 1
            knots[knot_index] = 0.5 * (points[i].x + points[i - 1].x)
            knot_index += 1

        spline.set_knots(knots)
        return True

def get_x(length):
    x = np.zeros(length)
    for i in range(length):
        x[i] = i + 1.0 / (i + 1) * np.random.rand()
    x = np.concatenate((x, x), 0)
    x = np.sort(x)
    return x


def get_data_1():
    x = get_x(60)
    n = len(x)
    y = [0] * n
    w = [1] * n
    for i in range(0, n, 2):
        y[i] = np.cos(0.2 * x[i])
        err = np.random.rand() * 15 / (x[i] + 10)
        y[i + 1] = y[i] + err
        y[i] -= err
        w[i] = 1.0 / math.fabs(x[i] - x[i - 1])
        w[i + 1] = w[i]

    return pnt.Points(x, y, w)


def get_data_2():
    x = get_x(10)
    n = len(x)
    y = [0] * n
    w = [1] * n
    for i in range(0, n, 2):
        y[i] = np.cos(0.2 * x[i])
        y[i] += 0.4 * np.cos(0.5 * x[i])
        err = np.random.rand() / 3
        y[i + 1] = y[i] + err
        y[i] -= err
        w[i] = 1.0 / math.fabs(x[i] - x[i - 1])
        w[i + 1] = w[i]

    return pnt.Points(x, y, w)


def get_data_3():
    x = get_x(60)
    n = len(x)
    y = [0] * n
    w = [1] * n
    for i in range(0, n, 2):
        y[i] = np.cos(0.005 * x[i] * x[i])
        err = np.random.rand() * 15 / (x[i] + 10)
        y[i + 1] = y[i] + err
        y[i] -= err
        w[i] = 1.0 / math.fabs(x[i] - x[i - 1])
        w[i + 1] = w[i]

    return pnt.Points(x, y, w)


def main():
    points = get_data_3()

    # calculate new x's and y's
    x_curve = np.linspace(points.x[0], points.x[-1], 1000)
    y_curve = [None] * 1000
    y_curve_opt = [None] * 1000

    knots = [points.x[0], 1, 2, 3, 4, points.x[len(points) - 1]]
    coefficients = [None] * (len(points) + 2)
    knots.append(knots[len(knots) - 2] + 1)
    knots.sort()
    s = splpckg.Spline(coefficients, knots, 3)
    c = PlotSpline(s)
    c.initiate_grid(s, points)
    q = 1e-9
    c.approximate(s, points, q)

    index = 0
    for point in x_curve:
        y_curve[index] = s.get_value(point)
        index += 1

    index = 0
    knots = s.get_knots()
    knots_y_uni = [0] * len(knots)
    for point in knots:
        knots_y_uni[index] = s.get_value(point)
        index += 1


    index = 0
    for point in x_curve:
        y_curve_opt[index] = s.get_value(point)
        index += 1

    index = 0
    knots2 = s.get_knots()
    knots_y = [0] * len(knots2)
    for point in knots2:
        knots_y[index] = s.get_value(point)
        index += 1

    plt.plot(x_curve, y_curve, points.x, points.y, 'o', knots, knots_y_uni, 's')
    plt.legend(["Smoothed spline", "Data", "Evenly spreaded knots"])
    plt.xlim([points.x[0] - 1, points.x[-1] + 1])
    plt.show()

main()
