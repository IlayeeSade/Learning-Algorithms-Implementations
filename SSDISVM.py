# imports
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


class SSDISVM:
  """
  Support Vector Machine Classifier, with SMO and more optimization algorithms by Swissa Sade David Ilayee (SSDI)
  """

  def __init__(self):
    self.kernel_type = None
    self.C = None
    self.tol = None
    self.eps = None
    self.gamma = None
    self.gammas = None
    self.poly_degree = None
    # Initialize for the fit method
    self.X = None  # x_examples
    self.Y = None  # y_examples
    self.__Ei = 0  # error of alpha_i
    self.__Ej = 0  # error of alpha_j
    self.__A = None  # alphas
    self.__W = None  # weights
    self.__b = 0  # bias
    # For grid
    self.red_positions = []
    self.blue_positions = []

  # Kernel Functions --START--
  @staticmethod
  def linear_kernel(x1, x2):
    """
    linear kernel function x1 @ x2.transpose()
    """
    return x1 @ x2.transpose()

  def polynomial_kernel(self, x1, x2):
    """
    polynomial kernel function (x1 @ x2.transpose()) ** poly_degree
    """
    return (self.gamma * x1 @ x2.transpose()) ** self.poly_degree

  def rbf_kernel(self, x1, x2):
    """
    rbf kernel function (Gaussian) exp(-gamma * ||x1 - x2||^2)
    """
    if x2.ndim == 1:
      if x1.ndim == 1:
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2, axis=0) ** 2)
      return np.exp(-self.gamma * np.linalg.norm(x1 - x2, axis=1) ** 2)
    return np.exp(-self.gamma * np.linalg.norm(x1[:, np.newaxis, :] - x2[np.newaxis, :, :], axis=2) ** 2)
    # this line has some manipulation to make it to make each row of x2 subtract every row of x1,
    # (m,#Features) - (inputs,#Features) ... (m,inputs,#Features)

  def sigmoid_kernel(self, x1, x2):
    """
    sigmoid kernel function tanh(gamma * x1 @ x2.transpose())
    """
    return np.tanh(self.gamma * x1 @ x2.transpose())

  # Kernel Functions --END--
  # Calculators/Helpers --START--

  def __calc_error_a(self, i):
    """
    Calculate the error of x_train[i] : f(x) - y
    """
    return self.g_a(self.X[i]) - self.Y[i] - self.__b

  def __calc_bounds(self, i, j):
    """
    calculate the L and H of the model with regularization/slack variables
    correct bounds for alpha j
    """
    y1 = self.Y[i]
    y2 = self.Y[j]
    alpha1 = self.__A[i]
    alpha2 = self.__A[j]
    if y1 != y2:
      return max(0, alpha2 - alpha1), min(self.C, self.C + alpha2 - alpha1)
    else:
      return max(0, alpha1 + alpha2 - self.C), min(self.C, alpha1 + alpha2)

  def __calc_eta(self, x1, x2):
    """
    calculate the eta of the model 2 * K(x1, x2) - K(x1, x1) - K(x2, x2)
    """
    return 2 * self.kernel_type(x1, x2) - self.kernel_type(x1, x1) - self.kernel_type(x2, x2)

  def __calc_obj(self):
    """
    Calculate the objective function W(alpha) = 1/2 * alpha @ A @ alpha - sum(alpha)
    To be improved...
    """
    return 1 / 2 * (self.Y.reshape(-1, 1) * self.g_a(self.X)).reshape(1, -1) @ self.__A - np.sum(self.__A) # To be improved...

  def __calc_b(self, i, j, alpha_i_old, alpha_j_old):
    """
    calculate the bias of the model with regularization/slack variables
    """
    var1 = self.Y[i] * (self.__A[i] - alpha_i_old)
    var2 = self.Y[j] * (self.__A[j] - alpha_j_old)
    var3 = self.kernel_type(self.X[i], self.X[j])
    b1 = (self.__Ei + var1 * self.kernel_type(self.X[i], self.X[i]) + var2 * var3 + self.__b)
    if 0 < self.__A[i] < self.C:
      self.__b = b1
      return
    b2 = (self.__Ej + var1 * var3 + var2 * self.kernel_type(self.X[j], self.X[j]) + self.__b)
    if 0 < self.__A[j] < self.C:
      self.__b = b2
      return
    self.__b = (b1 + b2) / 2
    return

  def __calc_weights(self):
    """
    calculate the weights of the model, works only for linear kernel
    """
    self.__W = np.dot(self.__A * self.Y * self.gamma, self.X)

  @staticmethod
  def clip_a(a, l, h):
    """
    clip the alpha value to be between l and h
    to fulfill the KKT conditions
    """
    if a > h:
      return h
    if a < l:
      return l
    return a

  # Calculators/Helpers --END--
  # Predictions --START--

  def g_a(self, x):
    """
        calculate the g(x) using alphas
        """
    # Reshape x to be a 2D array if it's 1D
    if x.ndim == 1:
      x = x.reshape(1, -1)
    return self.kernel_type(self.X, x).transpose() @ (self.__A * self.Y).reshape(-1, 1)

  def pred_a(self, x):
    """
    predict the model using alphas
    """
    return np.sign(self.g_a(x) - self.__b)

  def pred_w(self, x):
    """
    predict the model using weights
    To be constructed...
    """
    return np.sign(self.g_a(x) - self.__b) # To be constructed...

  # Predictions --END--
  # SMO Algorithm --START--

  def __step(self, i, j):
    """
    Take a step, update the alphas and the bias if loss function decreases
    and abs(a_j - alpha_j_old) < self.eps * (a_j + alpha_j_old + self.eps)
    """
    if i == j:
      return 0
    y_i = self.Y[i]
    y_j = self.Y[j]
    alpha_i_old = self.__A[i]
    alpha_j_old = self.__A[j]
    self._Ei = self.__calc_error_a(i)
    s = y_i * y_j
    l, h = self.__calc_bounds(i, j)
    eta = self.__calc_eta(self.X[i], self.X[j])
    if l == h:
      return 0
    if eta > 0:
      a_j = alpha_j_old - y_j * (self.__Ei - self.__Ej) / eta
      a_j = self.clip_a(a_j, l, j)
    else:
      self.__A[j] = l
      self.__A[i] = alpha_i_old + s * (alpha_j_old - self.__A[j])
      obj_l = self.__calc_obj()
      self.__A[j] = h
      self.__A[i] = alpha_i_old + s * (alpha_j_old - self.__A[j])
      obj_h = self.__calc_obj()
      if obj_l < obj_h - self.eps:
        a_j = l
      elif obj_l > obj_h + self.eps:
        a_j = h
      else:
        a_j = alpha_j_old
    if abs(a_j - alpha_j_old) < self.eps * (a_j + alpha_j_old + self.eps):
      return 0
    a_i = alpha_i_old + s * (alpha_j_old - a_j)
    self.__calc_b(i, j, alpha_i_old, alpha_j_old)
    self.__A[i] = a_i
    self.__A[j] = a_j
    return 1

  def examine_example(self, j):
    """
    check if you can take a step to optimize the model
    for this particular alpha_j, if you can, take it.
    """
    y_j = self.Y[j]
    alpha_j = self.__A[j]
    self.__Ej = self.__calc_error_a(j)
    r_j = self.__Ej * y_j
    if (r_j < -self.tol and alpha_j < self.C) or (r_j > self.tol and alpha_j > 0):
      if len(self.__A[(self.__A != 0) & (self.__A != self.C)]) > 1:
        if self.__Ei > 0:
          i = np.argmin(self.__A)  # Second choice heuristic, max abs(Ei - Ej)
        else:
          i = np.argmax(self.__A)  # Second choice heuristic, max abs(Ei - Ej)
        if self.__step(i, j):
          return 1
      for i in np.random.permutation(self.X.shape[0]):  # if the second choice heuristic fails check randomly,
        # in order to avoid bias towards the first examples in the dataset
        if self.__A[i] != 0 and self.__A[i] != self.C:
          if self.__step(i, j):
            return 1
      for i in np.random.permutation(self.X.shape[0]):  # if the second choice heuristic fails check randomly,
        # in order to avoid bias towards the first examples in the dataset
        if self.__step(i, j):
          return 1
    return 0

  def smo_with_regularization(self):
    """
        smo with regularization implementation
        """
    self.__A = np.zeros(self.X.shape[0])  # Initialize alphas at 0
    self.__b = 0  # Initialize bias at 0
    num_changed_alphas = 0
    examine_all = True
    while num_changed_alphas > 0 or examine_all:
      num_changed_alphas = 0
      if examine_all:
        for i in range(self.X.shape[0]):
          num_changed_alphas += self.examine_example(i)
      else:
        for i in range(self.X.shape[0]):
          if self.__A[i] != 0 and self.__A[i] != self.C:
            num_changed_alphas += self.examine_example(i)
      # A switcher to handle if improvements were not made in the expected alphas.
      # (Non bound alphas, more likely to change)
      if examine_all:
        examine_all = False
      elif num_changed_alphas == 0:
        examine_all = True

  # SMO Algorithm --END--
  # Fit Method --START--

  def fit(self, algo='smo_with_regularization', x = 'grid', y = 'grid'):
    """
    fit the model to the data
    """
    if x != 'grid':
      if y != 'grid':
        self.X = x
        self.Y = y
    algorithms = {
      # 'simple_smo_with_regularization': self.simple_smo_with_regularization, # (Was Implemented, sucked ass)
      # 'simple_smo': self.simple_smo, # (Was Implemented, sucked ass)
      # 'gradient_descent': self.gradient_descent, # (To be implemented...)
      'smo_with_regularization': self.smo_with_regularization
    }
    algorithms[algo]()
    self.__calc_weights()

  # Fit Method --END--
  # Plotting --START--
  def plot(self, method='alphas'):
    """
        plot the data and the decision boundary
        """
    methods = {
      'weights': self.pred_w,
      'alphas': self.pred_a
    }
    if self.X is not None and self.Y is not None:
      x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
      y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
      xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.04),
                           np.arange(y_min, y_max, 0.04))

      # Predict class labels for all points in the mesh
      Z = methods[method](np.c_[xx.ravel(), yy.ravel()])
      # Put the result into a color plot
      Z = Z.reshape(xx.shape)

      # Plot the decision boundary and the original data
      plt.contourf(xx, yy, Z, alpha=0.8, cmap='spring')
      plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, edgecolors='w', marker='o', cmap='PiYG')
      plt.title("SVM Decision Boundary Using: " + method.upper())
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.show()

  def print_info(self):
    """
    print the info of the model
    """
    print(f'alphas: {self.__A}, weights: {self.__W}, bias: {self.__b}, gamma: {self.gamma}')
    print(f'C: {self.C}, tol: {self.tol}, max_iters: {self.max_iters}, eps: {self.eps}')
    print(f'kernel_type: #fix, poly_degree: {self.poly_degree}')
    print(f'Examples Variances: {self.X.var(axis=0)}, Examples Means: {self.X.mean(axis=0)}')

  def loader(self, kernel_type='linear', C=1.0, tol=0.001, max_iters=2000, eps=0.001, gamma='scale', poly_degree=1):
    kernel_functions = {
      'linear': self.linear_kernel,
      'poly': self.polynomial_kernel,
      'rbf': self.rbf_kernel,
      'sigmoid': self.sigmoid_kernel
    }
    self.kernel_type = kernel_functions[kernel_type]
    self.C = C
    self.tol = tol
    self.max_iters = max_iters
    self.eps = eps
    self.gammas = {  # Setting the gamma
      'auto': 1 / self.X.shape[1],
      'scale': 1 / (self.X.shape[1] * self.X.var().mean())
    }
    if gamma not in self.gammas:
      self.gamma = gamma
    else:
      self.gamma = self.gammas[gamma]
    self.poly_degree = poly_degree

  def load_grid(self):
    red_array = np.array(self.red_positions)-10
    blue_array = np.array(self.blue_positions)-10
    print("Pink positions as a NumPy array:\n", red_array)
    print("Yellow positions as a NumPy array:\n", blue_array)
    self.X = np.concatenate([red_array, blue_array], axis=0)
    self.Y = np.concatenate([-np.ones(red_array.shape[0]), np.ones(blue_array.shape[0])], axis=0)

  def grid(self):
    # Set up the grid with a focus on the middle
    grid_size = 21  # Making it odd to have a clear center
    center = grid_size // 2
    grid = np.zeros((grid_size, grid_size))

    # Flag to check if the process is ended
    is_ended = False

    # Function to draw the grid
    def draw_grid():
        if is_ended:
            return

        clear_output(wait=True)  # Clear the previous plot
        fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better clarity

        # Set up the ticks and grid
        ticks = np.arange(-10, 11)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks[::-1])  # Reverse y-axis labels
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)

        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-10.5, 10.5)

        # Plot X's for clicked positions
        for pos in self.red_positions:
            ax.text(pos[1] - center, center - pos[0], 'X', ha='center', va='center', color='tab:pink', fontsize=15)
        for pos in self.blue_positions:
            ax.text(pos[1] - center, center - pos[0], 'X', ha='center', va='center', color='yellow', fontsize=15)

        plt.show()

        # Display the widgets again after clearing the output
        display(end_button)
        display(grid_ui)

    # Function to handle clicks
    def on_click(x, y):
        if is_ended:
            return

        position = (y, x)
        if position in self.red_positions:
            self.red_positions.remove(position)
            self.blue_positions.append(position)
        elif position in self.blue_positions:
            self.blue_positions.remove(position)
        else:
            self.red_positions.append(position)
        draw_grid()

    # Function to end the process
    def end_process(b):
        is_ended = True
        clear_output(wait=True)
        plt.close('all')  # Close all open plots
        print("Process ended. Final positions:")
        self.load_grid()

    # Create buttons for each cell in the grid
    buttons = []
    for y in range(grid_size):
        button_row = []
        for x in range(grid_size):
            button = widgets.Button(description=f'({x-center},{center-y})',
                                    layout=widgets.Layout(width='60px', height='40px'),
                                    style={'font_size': '10px'})
            button.on_click(lambda b, x=x, y=y: on_click(x, y))
            button_row.append(button)
        buttons.append(widgets.HBox(button_row))

    # Create an "End Process" button
    end_button = widgets.Button(description="End Process",
                                layout=widgets.Layout(width='200px', height='50px'),
                                style={'button_color': 'lightcoral'})
    end_button.on_click(end_process)

    # Display buttons in a grid layout, centered on (0,0)
    grid_ui = widgets.VBox(buttons)

    # Initial display of the end button and grid
    display(end_button)
    display(grid_ui)

    # Initial draw of the grid
    draw_grid()
