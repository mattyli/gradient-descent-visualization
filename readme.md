# Gradient Descent Visualization

This notebook helps visualize gradient descent, assuming a loss function represented as a 3D surface. Currently I support `lr` and `steps` (learning rate and steps), and in the future I hope to add support for different optimizers and etc.

Surfaces are implemented as python classes and can be found in `surfaces.py`. Gradients are manually derived from the original function
for each different surface (good calculus practice). 

Plotly is used to generate and animate 3D visuals.

![Visuals](sgd_quadratic.gif)

