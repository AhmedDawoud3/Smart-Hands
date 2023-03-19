# Smart-Hands
Smart-Hands is a project that demonstrates how to implement a simple backpropagation algorithm to control the angles of a robotic hand.

<img src="https://user-images.githubusercontent.com/68483546/226207464-20d73f58-339d-48e2-99c5-b1c1a3a9799c.gif" width="535" height="299"/>

## How it works:
- By keeping track of each angle's value as an independent variable, 
- Calculated the loss using the squared distance between the tip of the last hand and the target ("Denoted by the yellow line")
- Calculating the local derivatives of each's wrt the loss
- Using the calculated gradient, optimizing the loss as a function of the angles

### The project visualizes the hand and the target using love2d.
