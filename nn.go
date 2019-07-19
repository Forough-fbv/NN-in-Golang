package nn

import "github.com/gonum/v1/gonum/mat"

// forward takes input data and returns the 'activation' and 'z'
// from each layer - z = w.x +b and a = sigmoid(z)
func (n *MLP) forward(x mat.Matrix) (as, zs []mat.Matrix) {
	as = append(as, x) // first activation is input

	_x := x

	for i := 0; i < len(n.weights); i++ {
		w := n.weights[i]
		b := n.biases[i]

		dot := new(mat.Dense)
		dot.Mul(_x, w)

		z := new(mat.Dense)
		addB := func(_, col int, v float64) float64 { return v + b.At(col, 0) }
		z.Apply(addB, dot)

		a := new(mat.Dense)
		a.Apply(applySigmoid, z)

		zs = append(zs, z)
		as = append(as, a)

		_x = a
	}

	return
}
