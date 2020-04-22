package mleap

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"archive/zip"
	"log"
	"math"
	"io/ioutil"
	"strings"
	"encoding/json"
	"errors"
)

type ModelLoader interface {
	Load(filePath string) *MLModel
}

type WordToVecModelLoader struct { }

type MLModel interface {
	Transform(textInput []string) *mat.VecDense
}

type WordToVecModel struct {
	VectorMap map[string]*mat.VecDense
	Norms map[string]float64
	VectorLength int
}

func (model *WordToVecModel) Transform(sentence []string) *mat.VecDense {
	sumRaw := make([]float64, model.VectorLength)
	sumVec := mat.NewVecDense(model.VectorLength, sumRaw)
	
	var sentenceLen = len(sentence)
	for _, token := range sentence {
		wordVector, vectorFound := model.VectorMap[token]
		if (vectorFound) {
			sumVec.AddVec(sumVec, wordVector)
		}
	}

	sumVec.ScaleVec(1/float64(sentenceLen), sumVec)

	return sumVec
}


func (model *WordToVecModel) Distance(token1, token2 string) (float64, error) {
	vector1, vector1Found := model.VectorMap[token1]
	if (!vector1Found) {
		return 0.0, errors.New(fmt.Sprintf("No vector found for token %s", token1))
	}
	
	vector2, vector2Found  := model.VectorMap[token2]
	if (!vector2Found) {
		return 0.0, errors.New(fmt.Sprintf("No vector found for token %s", token2))
	}

	norm1, norm1Found := model.Norms[token1]
	if (!norm1Found) {
		return 0.0, errors.New(fmt.Sprintf("No norm found for token %s", token1))
	}
	
	norm2, norm2Found := model.Norms[token2]
	if (!norm2Found) {
		return 0.0, errors.New(fmt.Sprintf("No norm found for token %s", token2))
	}
	
	dotProduct := mat.Dot(vector1, vector2)
	cosineSim := dotProduct / (norm1 * norm2)
	return cosineSim, nil
}

type WordToVecJson struct {
	Attributes Attributes `json:"attributes"`
	Op string             `json:"op"`
}

type Attributes struct {
	Words Words `json:"words"`
	Indices Indices `json:"indices"` 
	WordVectors WordVectors `json:"word_vectors"`
}

type Words struct {
	Tokens []string `json:"string"`
	Type string   `json:"type"`
}

type Indices struct {
	Cursors []int64 `json:"long"`
	Type string `json:"type"`
}

type WordVectors struct {
	Vectors []float64 `json:"double"`
	Type string    `json:"type"`
}

func (loader *WordToVecModelLoader) Load(filePath string) (*WordToVecModel, error) {
	reader, err := zip.OpenReader(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer reader.Close()

	var model *WordToVecModel = nil
	
	for _, file := range reader.File {
		fmt.Printf("Contents of %s:\n", file.Name)
		if strings.Contains(file.Name,"model.json") {
			rc, err := file.Open()

			if (err!=nil) {
				return nil, err
			}

			content, err := ioutil.ReadAll(rc)

			if (err!=nil) {
				return nil, err
			}

			var modelStruct WordToVecJson
			json.Unmarshal([]byte(content), &modelStruct)

			modelAttr := modelStruct.Attributes
			
			var curLength = len(modelAttr.Indices.Cursors)
			var vectorsLength = len(modelAttr.WordVectors.Vectors)
			var dimSize = vectorsLength / curLength

			vectorMap := make(map[string]*mat.VecDense)
			norms := make(map[string]float64)

			for idx := 0; idx < len(modelAttr.Words.Tokens); idx++ {
				token := modelAttr.Words.Tokens[idx]
				vecStartIdx := int(modelAttr.Indices.Cursors[idx])*dimSize
				vecEndIdx := vecStartIdx + dimSize
				tokenVector := modelAttr.WordVectors.Vectors[vecStartIdx:vecEndIdx]
				tokenDense := mat.NewVecDense(dimSize, tokenVector)
				vectorMap[token]=tokenDense
				tokenNorm := math.Sqrt(mat.Dot(tokenDense, tokenDense))
				norms[token]=tokenNorm
			}
			
			model=&WordToVecModel{vectorMap, norms, dimSize}			
			rc.Close()
			return model, nil
		}		
	}

	return model, nil
}
