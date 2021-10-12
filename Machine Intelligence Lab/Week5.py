import numpy as np


class Tensor:

    """
    Tensor Wrapper for Numpy arrays.
    Implements some binary operators.
    Array Broadcasting is disabled
    Args:
        arr: Numpy array (numerical (int, float))
        requires_grad: If the tensor requires_grad (bool)(otherwise gradient dont apply to the tensor)
    """

    def __init__(self, arr, requires_grad=True):

        self.arr = arr
        self.requires_grad = requires_grad

        # When node is created without predecessor the op is denoted as 'leaf'
        # 'leaf' signifies leaf node
        self.history = ['leaf', None, None]
        # History stores the information of the operation that created the Tensor.
        # Check set_history

        # Gradient of the tensor
        self.zero_grad()
        self.shape = self.arr.shape

    def zero_grad(self):
        """
        Set grad to zero
        """
        self.grad = np.zeros_like(self.arr)

    def set_history(self, op, operand1, operand2):
        """
        Set History of the node, indicating how the node was created.
        Ex:-
            history -> ['add', operand1(tensor), operand2(tensor)]
            history -> ['leaf', None, None] if tensor created directly
        Args:
            op: {'add', 'sub', 'mul', 'pow', 'matmul', 'leaf') (str)
            operand1: First operand to the operator. (Tensor object)
            operand2: Second operand to the operator. (Tensor object)
        """
        self.history = []
        self.history.append(op)
        self.requires_grad = False
        self.history.append(operand1)
        self.history.append(operand2)

        if operand1.requires_grad or operand2.requires_grad:
            self.requires_grad = True

    """
    Addition Operation
    Tensor-Tensor(Element Wise)
    __add__: Invoked when left operand of + is Tensor
    grad_add: Gradient computation through the add operation
    """

    def __add__(self, other):
        """
        Args:
            other: The second operand.(Tensor)
                    Ex: a+b then other -> b, self -> a
        Returns:
            Tensor: That contains the result of operation
        """
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                raise ArithmeticError(
                    f"Shape mismatch for +: '{self.shape}' and '{other.shape}' ")
            out = self.arr + other.arr
            out_tensor = Tensor(out)
            out_tensor.set_history('add', self, other)

        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'")

        return out_tensor

    """
    Matrix Multiplication Operation (@)
    Tensor-Tensor
    __matmul__: Invoked when left operand of @ is Tensor
    grad_matmul: Gradient computation through the matrix multiplication operation
    """

    def __matmul__(self, other):
        """
        Args:
            other: The second operand.(Tensor)
                    Ex: a+b then other -> b, self -> a
        Returns:
            Tensor: That contains the result of operation
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"unsupported operand type(s) for matmul: '{self.__class__}' and '{type(other)}'")
        if self.shape[-1] != other.shape[-2]:
            raise ArithmeticError(
                f"Shape mismatch for matmul: '{self.shape}' and '{other.shape}' ")
        out = self.arr @ other.arr
        out_tensor = Tensor(out)
        out_tensor.set_history('matmul', self, other)

        return out_tensor

    def grad_add_helper(self, gradients=None):
        if(self.history[1].requires_grad):
            for i in range(self.arr.shape[0]):
                for j in range(self.arr.shape[1]):
                    self.history[1].grad[i][j] += self.history[1].arr[i][j] / self.history[1].arr[i][j]
        
        if(self.history[2].requires_grad):
            for i in range(self.arr.shape[0]):
                for j in range(self.arr.shape[1]):
                    self.history[2].grad[i][j] += self.history[2].arr[i][j] / self.history[2].arr[i][j]

        return((self.history[1].grad, self.history[2].grad))


    def grad_add(self, gradients=None):
        """
        Find gradients through add operation
        gradients: Gradients from successing operation. (numpy float/int)
        Returns:
            Tuple: (grad1, grad2)
            grad1: Numpy Matrix or Vector(float/int) -> Represents gradients passed to first operand
            grad2: Numpy Matrix or Vector(float/int) -> Represents gradients passed to second operand
            Ex:
                c = a+b
                Gradient to a and b
        """
        # TODO

        return(gradients, gradients)

        # DONE
        # pass

    def degrades(self, gradients=None):
        degrades_une_213 = np.ones_like(self.history[1].arr)
        degrades_deux_213 = np.ones_like(self.history[2].arr)

        degrades_une_213 = gradients @ np.transpose(self.history[2].arr)
        degrades_deux_213 = np.transpose(self.history[1].arr) @ gradients

        return(degrades_une_213, degrades_deux_213)


    def grad_matmul(self, gradients=None):
        """
        Find gradients through matmul operation
        gradients: Gradients from successing operation. (numpy float/int)
        Returns:
            Tuple: (grad1, grad2)
            grad1: Numpy Matrix or Vector(float/int) -> Represents gradients passed to first operand
            grad2: Numpy Matrix or Vector(float/int) -> Represents gradients passed to second operand
            Ex:
                c = a@b
                Gradients to a and b
        """
        # TODO

        return((self.degrades(gradients)))

        # DONE
        # pass

    def backward(self, gradients=None):
        """
        Backward Pass until leaf node.
        Setting the gradient of which is the partial derivative of node(Tensor) 
        the backward in called on wrt to the leaf node(Tensor).
        Ex:
            a = Tensor(..) #leaf
            b = Tensor(..) #leaf
            c = a+b
            c.backward()
            computes:
                dc/da -> Store in a.grad if a requires_grad
                dc/db -> Store in b.grad if b requires_grad
        Args:
            gradients: Gradients passed from succeeding node
        Returns:
            Nothing. (The gradients of leaf have to set in their respective attribute(leafobj.grad))
        """
        # TODO

        try:
            if(gradients == None):
                gradients = np.ones_like(self.arr)
        except:
            pass

        if(self.history[0] == 'leaf'):
            if(self.requires_grad == True):
                self.grad += gradients

        elif(self.history[0] == 'add'):
            self.history[1].backward(self.grad_add(gradients)[0])
            self.history[2].backward(self.grad_add(gradients)[1])

        elif(self.history[0] == 'matmul'):
            self.history[1].backward(self.grad_matmul(gradients)[0])
            self.history[2].backward(self.grad_matmul(gradients)[1])

        # DONE
        # pass
        

'''
a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
b = Tensor(np.array([[3.0, 2.0], [1.0, 5.0]]), requires_grad=False)
c = Tensor(np.array([[3.2, 4.5], [6.1, 4.2]]))
z = np.array([[0.0, 0.0], [0.0, 0.0]])
sans = a+b
sans2 = a+a
mulans = a@b
mulans2 = (a+b)@c
sgrad = np.array([[1.0, 1.0], [1.0, 1.0]])
sgrad2 = np.array([[2.0, 2.0], [2.0, 2.0]])
mulgrad = np.array([[5.0, 6.0], [5.0, 6.0]])
mulgrad2 = np.array([[4.0, 4.0], [6.0, 6.0]])
mulgrad3 = np.array([[7.7, 10.29], [7.7, 10.29]])
mulgrad4 = np.array([[8.0, 8.0], [13.0, 13.0]])


def test_case():

    try:
        sans.backward()
        np.testing.assert_array_almost_equal(a.grad, sgrad, decimal=2)
        print("Test Case 1 for the function Add Grad PASSED")
    except:
        print("Test Case 1 for the function Add Grad FAILED")

    try:
        np.testing.assert_array_almost_equal(b.grad, z, decimal=2)
        print("Test Case 2 for the function Add Grad PASSED")
    except:
        print("Test Case 2 for the function Add Grad FAILED")

    a.zero_grad()
    b.zero_grad()

    try:
        sans2.backward()
        np.testing.assert_array_almost_equal(a.grad, sgrad2, decimal=2)
        print("Test Case 3 for the function Add Grad PASSED")
    except:
        print("Test Case 3 for the function Add Grad FAILED")

    a.zero_grad()
    b.zero_grad()

    try:
        mulans.backward()
        np.testing.assert_array_almost_equal(a.grad, mulgrad, decimal=2)
        print("Test Case 4 for the function Matmul Grad PASSED")
    except:
        print("Test Case 4 for the function Matmul Grad FAILED")

    try:
        np.testing.assert_array_almost_equal(b.grad, z, decimal=2)
        print("Test Case 5 for the function Matmul Grad PASSED")
    except:
        print("Test Case 5 for the function Matmul Grad FAILED")

    a.zero_grad()
    b.zero_grad()
    b.requires_grad = True

    try:
        mulans.backward()
        np.testing.assert_array_almost_equal(b.grad, mulgrad2, decimal=2)
        print("Test Case 6 for the function Matmul Grad PASSED")
    except:
        print("Test Case 6 for the function Matmul Grad FAILED")

    a.zero_grad()
    b.zero_grad()
    c.zero_grad()

    try:
        mulans2.backward()
        np.testing.assert_array_almost_equal(a.grad, mulgrad3, decimal=2)
        np.testing.assert_array_almost_equal(b.grad, mulgrad3, decimal=2)
        print("Test Case 7 for the function Matmul and add Grad PASSED")
    except:
        print("Test Case 7 for the function Matmul and add Grad FAILED")

    try:
        np.testing.assert_array_almost_equal(c.grad, mulgrad4, decimal=2)
        print("Test Case 8 for the function Matmul and add Grad PASSED")
    except:
        print("Test Case 8 for the function Matmul and add Grad FAILED")

test_case()
'''
