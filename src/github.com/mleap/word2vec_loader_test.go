package mleap

import(
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func TestWord2VecLoader(t *testing.T) {
	wordToVecLoader := WordToVecModelLoader{}
	wordToVecModel, _ := wordToVecLoader.Load("/tmp/occ-word2vec-model-2020-01-16-12.zip")

	queryVec := wordToVecModel.Transform([]string{"scala","scala"})
	
}
