There are 3 agents you can work with:

1. Value Iteration agent:
	For this, you can use the file "train_VI.py". 
	You can run this by:
		python train_VI.py grid_configs/A1_grid.npy --sigma 0.0 --fps 5
		OR for the bigger grid:
		python train_VI.py grid_configs/hardex_grid.npy --sigma 0.0 --fps 5
	The choice of fps is to make the speed slower so you can see the agent's movement.

2. Q-Learning Agent:
	For this, you can run the "train_q_learning.py" file.
	At the beginning of the file, you can change the path for the grid, the number of episodes, and other 	parameters. 

3. Monte Carlo Agent:
	For this, you can run the "train_mc.py" file.
	At the beginning of the file, you can change the path for the grid, the number of episodes, and other 	parameters. 


In case of running into errors about not finding the agent, check the import part in the train files and make sure it follows the structure of folders/files on your computer. 

You can experiment with different values of sigma for value iteration and plot the different rewards for them, using the "sigma_testing.py": python "sigma_testing.py"
