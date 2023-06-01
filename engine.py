import numpy as np

class Variable:
    def __init__(self, data, label='', grad=0, _op='', _prev=()):
        self.data = data
        self.label = label
        self.grad = grad
        self._prev = _prev
        self._op = _op
        self._backward = lambda: None

    def backward(self):
        """
        The value stored in this Variable is the result of an entire computational graph.
        If this variable is L, dL/dL = 1

        This variable consists of two components. Backpropagate the gradient to each of the components.
        Make sure to call the components' backward(), too.

        If you simply call this Variable's backward(), and rely on backward() to recursively call backward(),
        the problem is you might double count gradients.

        Assume the following computational graph, where c = f(a, b), d = g(c), and e = h(c)

        d     e
         \   /
          \ /
           c
           /\
          /  \
         a    b
        The gradients from d and e will flow into c, and then the gradient from c will flow into a and b.
        We only want the gradient from c to flow into a and b once.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1

        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self):
        self.grad = 0
    
    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other, f'{other}')
        out = Variable(self.data + other.data, grad=0, _op='+', _prev=(self, other))
        
        def _backward():
            """
            '+' distributes its gradient to its components
            """
            # f = x + y
            # dfdx = 1
            # dfdy = 1
            # dldx = dldf * dfdx
            #      = grad * 1
            # dldy = dldf * dfdy
            #      = grad * 1
            # f
            self.grad += out.grad
            other.grad += out.grad
            
            # WARNING: Leave the below commented out. With the current implementation
            # backward will be called multiple times on a node, which will backpropagate
            # the gradients unnecessarily
            # self._backward()
            # other._backward()

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other, f'{other}')
        out = Variable(self.data - other.data, label=self.label, grad=0, _op='-', _prev=(self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += -1 * out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other, f'{other}')
        label = f'{self.label} * {other.label}'
        out = Variable(self.data * other.data, label=self.label, grad=0, _op='*', _prev=(self, other))
        
        def _backward():
            """
            f = a * b
            df/da = b
            df/db = a
            """
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out
    
    def __pow__(self, exponent):
        out = Variable(self.data ** exponent, label=self.label, grad=0, _op='pow', _prev=(self,))
        
        def _backward():
            """
            f = x^a
            df/dx = ax^(a - 1)
            """
            self.grad += out.grad * exponent * (self.data**(exponent - 1))

        out._backward = _backward
        return out

    def relu(self):
        out = Variable(data=max(0, self.data), label=self.label, grad=0, _op='relu', _prev=(self,))

        def _backward():
            """
            0 for self.data <= 0
            f(x) = x if x > 0
                 = 0 if x <= 0
            
            df/dx = 1
            df/dx = 0
            """
            if self.data <= 0:
                self.grad += 0
            else:
                self.grad += out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        return Variable(data=np.tanh(self.data), label=self.label, grad=0, _op='tanh', _prev=(self,))

    def __str__(self):
        if self._prev:
            labels = [p.label for p in self._prev]
            label = ','.join(labels)
        else:
            label = 'None'
        return f'V(value={self.data}, grad={self.grad}, _prev={label})'

    def __repr__(self):
        if self._prev:
            labels = [p.label for p in self._prev]
            label = ','.join(labels)
        else:
            label = 'None'
        return f'V(data={self.data}, grad={self.grad}, _prev={label})'