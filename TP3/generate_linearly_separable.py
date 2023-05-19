import random
from typing import Callable
import matplotlib.pyplot as plt



def generate_points_linearly_separable(x_range: (int, int) = (0, 5),
                                       y_range: (int, int) = (0, 5),
                                       n: int = 100,
                                       wrong: bool = False,
                                       f: Callable[[int], int] = (lambda x: x)) -> ([(float, float)], [(float, float)]):
    category_one = []
    category_minus_one = []
    # No dice que tiene que estar balanceado
    # si asi fuese se puede agregar una condicion medio boluda
    # si se demora mucho se puede hacer en n repes con un linspace sobre la curva y despues un
    # random entre el punto - el rango, pero me da fiaca codear eso,
    # asi que no optimicen prematuramente
    for i in range(n):
        
        new_point, w = get_new_point(x_range,y_range,f,wrong)
        if(not w):
            if is_point_greater_than(f=f, point=new_point):
                category_one.append(new_point)
            else:
                category_minus_one.append(new_point)
        else:
            if is_point_greater_than(f=f, point=new_point):
                category_minus_one.append(new_point)
            else:
                category_one.append(new_point)
        
    return category_one, category_minus_one

def get_new_point(x_range,y_range,f,wrong):
    new_point = (random.uniform(*x_range), random.uniform(*y_range))
    delta = f(new_point[0]) - new_point[1]
    condition = abs(delta) < 0.5
    if(wrong and condition):
        return new_point, wrong
    
    while condition:
        new_point = (random.uniform(*x_range), random.uniform(*y_range))
        delta = f(new_point[0]) - new_point[1]
        condition = abs(delta) < 0.5
    return new_point, False

# Greater than implica que el punto esta a la derecha de la linea
def is_point_greater_than(f: Callable[[int], int], point: (int, int)):
    # https://math.stackexchange.com/questions/324589/detecting-whether-a-point-is-above-or-below-a-slope
    return f(point[0]) > point[1]

if __name__ == '__main__':
    category_one, category_minus_one = generate_points_linearly_separable(f=lambda x: x)
    plt.scatter(*zip(*category_one), color='red')
    plt.scatter(*zip(*category_minus_one), color='blue')
    plt.show()