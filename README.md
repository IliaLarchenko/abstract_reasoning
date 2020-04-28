My solution for kaggle abstract reasoning competition.

Solution logic:
- preprocessing.py includes several functions starting with "get_". Each one of them takes the image (and sometimes additional parameters) as input and returns tuple (0, new_image) or (error_code, None).



Colors abstraction:
- Each color on each image is represented in several ways:
-- Absolute value (1 - is always 1, 2 - always 2 ...)
-- Presence on the image: most popular color, second popular colour, ... least popular.
-- The colour of the grid
