from numbers import Number


class VecBase:
    def __init__(self, hilbert, *args, **kwargs):
        self.hilbert = hilbert
        # save arguments for later use
        self.args = args
        self.kwargs = kwargs

    def scalar_multiplication(self, s):
        raise NotImplementedError

    def __mul__(self, other):
        if self.hilbert.is_scalar(other):
            return self.scalar_multiplication(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if self.hilbert.is_scalar(other):
            return self.scalar_multiplication(other)
        else:
            return NotImplemented


class Vec(VecBase):

    def get_ket(self):
        return self.hilbert.ket(*self.args, **self.kwargs)

    def get_bra(self):
        return self.hilbert.bra(*self.args, **self.kwargs)

    bra = property(get_bra)
    ket = property(get_ket)

    def __add__(self, other):
        # dont use isinstance(other, self.__class__) because
        # Ket + Bra can not be added!
        if other.__class__ == self.__class__:
            return self.hilbert.add_vectors(self, other)
        else:
            return NotImplemented



class Ket(Vec):

    def dual(self):
        return self.hilbert.bra(*self.args, **self.kwargs)



class Bra(Vec):

    def dual(self):
        return self.hilbert.ket(*self.args, **self.kwargs)

    def __mul__(self, other):
        if isinstance(other, self.hilbert.Ket_class):
            return self.hilbert.inner_product(self, other)
        else:
            return super().__mul__(other)


class Operator(VecBase):

    def dagger(self):
        raise NotImplementedError

    def scalar_multiplication(self, s):
        raise NotImplementedError

    def apply(self, ket_vec):
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, self.hilbert.Ket_class):
            return self.apply(other)
        elif isinstance(other, self.hilbert.Operator_class):
            return self.hilbert.mul_operators(self, other)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, self.hilbert.Bra_class):
            return (self.dagger() * other.dual()).dual()
        else:
            return super().__rmul__(other)

    def __add__(self, other):
        if other.__class__ == self.__class__:
            return self.hilbert.add_operators(self, other)
        elif self.hilbert.is_scalar(other):
            return self.hilbert.add_operators(self, other * self.hilbert.unit_operator())
        else:
            return super().__add__(other)            

    def __radd__(self, other):
        if self.hilbert.is_scalar(other):
            return self.hilbert.add_operators(other * self.hilbert.unit_operator(), self)
        else:
            return NotImplemented



class Hilbert:

    Vec_class = Vec
    Ket_class = Ket
    Bra_class = Bra
    Operator_class = Operator
    
    def inner_product(self, v, w):
        raise NotImplementedError

    def is_scalar(self, s):
        # The field for the vector space
        raise NotImplementedError

    def vec(self, *args, **kwargs):
        return self.Vec_class(self, *args, **kwargs)

    def bra(self, *args, **kwargs):
        return self.Bra_class(self, *args, **kwargs)

    def ket(self, *args, **kwargs):
        return self.Ket_class(self, *args, **kwargs)

    def operator(self, *args, **kwargs):
        return self.Operator_class(self, *args, **kwargs)

    def unit_operator(self):
        raise NotImplementedError

    def mul_operators(self, A, B):
        raise NotImplementedError

    def add_operators(self, A, B):
        raise NotImplementedError

    def add_vectors(self, v, w):
        raise NotImplementedError








# ---------------------------------------------

class DummyVec(Vec):
    def __init__(self, hilbert, name):
        super().__init__(hilbert, name)
        self.name = name

    def scalar_multiplication(self, s):
        return self.__class__(self.hilbert, "{} * {}".format(s, self.name))

class DummyBra(Bra, DummyVec):
    def __repr__(self):
        return "<{}|".format(self.name)

class DummyKet(Ket, DummyVec):
    def __repr__(self):
        return "|{}>".format(self.name)

class DummyOperator(Operator):
    def __init__(self, hilbert, name):
        super().__init__(hilbert, name)
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def dagger(self):
        return self.hilbert.operator(name="{}+".format(self.name))

    def scalar_multiplication(self, s):
        return self.__class__(self.hilbert, "{} * {}".format(s, self.name))

    def apply(self, ket_vec):
        return self.hilbert.ket(name="{} {}".format(self.name, ket_vec.name))


class DummyHilbert(Hilbert):

    Vec_class = DummyVec
    Ket_class = DummyKet
    Bra_class = DummyBra
    Operator_class = DummyOperator

    def is_scalar(self, s):
        return isinstance(s, Number)

    def inner_product(self, v, w):
        return "<{}|{}>".format(v.name, w.name)

    def mul_operators(self, A, B):
        return self.operator(name="({} * {})".format(A.name, B.name))

    def add_operators(self, A, B):
        return self.operator(name="({} + {})".format(A.name, B.name))

    def add_vectors(self, v, w):
        res = self.vec(name="({} + {})".format(v.name, w.name))
        if isinstance(v, self.Ket_class):
            return res.ket
        elif isinstance(v, self.Bra_class):
            return res.bra
        else:
            return res

    def unit_operator(self):
        return self.operator("I")










if __name__ == '__main__':
    H = DummyHilbert()
    psi = H.vec("psi")
    phi = H.vec("phi")
    A = H.operator("A")
    B = H.operator("B")

    print(psi.bra * A * phi.ket)
    phi2 = 2 * phi
    print(psi.bra * phi2.ket)
    psiPi = 3.14 * psi
    print(psiPi.bra * phi2.ket)
    print(12 * psi.bra * 3 * 10 * phi.ket)

    print(psi.ket + phi.ket)
    print((psi + phi).ket)
    print( A * B * phi.ket)
    print( phi.bra * A * B)
    print( (A * B) * phi.ket)
    print( 2 * A * phi.ket)
    print( (2 + A) * phi.ket)

    try:
        print(phi * psi.bra)
    except TypeError:
        print("Correctly thrown type error")

    try:
        print(phi + psi.bra)
    except TypeError:
        print("Correctly thrown type error")

    try:
        print( 2 + A * phi.ket)
    except TypeError:
        print("Correctly thrown type error")        