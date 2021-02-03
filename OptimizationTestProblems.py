# --------------------------------------------------------
# This script is used to test optimization algorithm
# --------------------------------------------------------
import numpy as np
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting


def Hartmann6(x):
    """Hartmann6 function (6-dimensional with 1 global minimum and 6 local minimum)
    minimums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    fmin = -3.32237
    fmax = 0.0 
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j,k] * ((x[k] - P[j,k]) ** 2)
        y -= alpha_j * np.exp(-t)

    return y

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    x: vector of input values
    x should be restricted in [-32,32]
    minimums = [0,0,0...]
    fmin = 0
    """
    d = len(x) # dimension of input vector x
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

#######################################################
#test ax
if __name__ == '__main__': 
    def homemade(x):
        "10,6,2,3,6,0"
        return (x[0]-10)**2+(x[1]-6)**2+(x[2]-2)**2+(x[3]-3)**2+(x[4]-6)**2+x[5]**2

    def hartmann_evaluation_function(parameterization):
        x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
        # In our case, standard error is 0, since we are computing a synthetic function.
        return {"hartmann6": (homemade(x), 0.0)}
        
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [-15, 15],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [-15, 15],
            },
            {
                "name": "x3",
                "type": "range",
                "bounds": [-15, 15],
            },
            {
                "name": "x4",
                "type": "range",
                "bounds": [-15, 15],
            },
            {
                "name": "x5",
                "type": "range",
                "bounds": [-15, 15],
            },
            {
                "name": "x6",
                "type": "range",
                "bounds": [-15, 15],
            },
        ],
        experiment_name="test",
        objective_name="hartmann6",
        evaluation_function=hartmann_evaluation_function,
        minimize=True,  # Optional, defaults to False.
        #parameter_constraints=["x1 + x2 <= 20"],  # Optional.
        total_trials=50, # Optional.
    )
    print('best_parameters',best_parameters)
    print('means',values)