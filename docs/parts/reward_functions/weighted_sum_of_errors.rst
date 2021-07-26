Weighted Sum of Errors
######################

Usage Guide
***********

To use the weighted sum of errors, you have to import the class, initialize an object and pass it to the environment.


.. code-block:: python

    import gym_electric_motor as gem
    from gym_electric_motor.reward_functions import WeightedSumOfErrors


    # initialize the reward function
    wse = WeightedSumOfErrors(
        reward_weights=dict(i_a=1, i_e=2) # Current control problem. Tracking of i_e is rewarded better.
        reward_power=2 # Squared Error
        # Alternative: reward_power=dict(i_a=1, i_e=0.5) Absolute error i_a, root error on i_e
        bias='positive' # Shift the reward range from negative to positive
        violation_reward=-250 # Self defined violation reward
        gamma=0.9 # Ignored, if a violation_reward is defined.
        normed_reward_weights=False # Otherwise weights will be normed automatically to sum up to 1.

    )

    # pass it to the environment
    env = gem.make('my-env-id-v0', reward_function=wse)

API Documentation
*****************

.. autoclass:: gym_electric_motor.reward_functions.weighted_sum_of_errors.WeightedSumOfErrors
   :members:
   :inherited-members:
