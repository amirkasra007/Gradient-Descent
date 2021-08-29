from numpy import *

# y = w0 + w1*x1 + w2*x2

def compute_error_for_line_given_points(w, points):
    
    totalError = 0
    for i in range(0, len(points)):
        x1 = points[i, 0]
        x2 = points[i, 1]
        y = points[i, 2]
        totalError += (y - (w[1] * x1 + w[2] * x2 + w[0])) ** 2
    return totalError / float(len(points))



def step_gradient(w_current, points, learningRate):

    w_gradient = [0, 0, 0]
    new_w = [0, 0, 0]
    N = float(len(points))
    for i in range(0, len(points)):
        x1 = points[i, 0]
        x2 = points[i, 1]
        y = points[i, 2]
        w_gradient[0] += -(2/N) * (y - ((w_current[1] * x1) + (w_current[2] * x2) + w_current[0]))
        w_gradient[1] += -(2/N) * x1 * (y - ((w_current[1] * x1) + (w_current[2] * x2) + w_current[0]))
        w_gradient[2] += -(2/N) * x2 * (y - ((w_current[1] * x1) + (w_current[2] * x2) + w_current[0]))
    new_w[0] = w_current[0] - (learningRate * w_gradient[0])
    new_w[1] = w_current[1] - (learningRate * w_gradient[1])
    new_w[2] = w_current[2] - (learningRate * w_gradient[2])
    return new_w



def gradient_descent_runner(points, starting_w, learning_rate, num_iterations):
    w = starting_w
    for i in range(num_iterations):
        w = step_gradient(w, array(points), learning_rate)
    return w


def run():

    points = genfromtxt("E:\Python\Gradient descent\data.csv", delimiter=",")
    learning_rate = 0.1 #alpha
    initial_w = [0, 0, 0]
    num_iterations = 100
    print ("Starting gradient descent at w0 = {0}, w1 = {1}, w2 = {2}, error = {3}".format(initial_w[0], initial_w[1], initial_w[2], compute_error_for_line_given_points(initial_w, points)))
    print ("Running...")
    w = gradient_descent_runner(points, initial_w, learning_rate, num_iterations)
    print ("After {0} iterations w0 = {1}, w1 = {2}, w2 = {3}, error = {4}".format(num_iterations, w[0], w[1], w[2], compute_error_for_line_given_points(w, points)))

if __name__ == '__main__':
    run()

