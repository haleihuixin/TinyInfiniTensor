#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Tensor A = inputs[0];
        Tensor B = inputs[1];
        Shape A_dim = A->getDims();
        Shape B_dim = B->getDims();

        if (transA && A_dim.size() >= 2) {
            std::swap(A_dim[A_dim.size() - 1], A_dim[A_dim.size() - 2]);
        }
        if (transB && B_dim.size() >= 2) {
            std::swap(B_dim[B_dim.size() - 1], B_dim[B_dim.size() - 2]);
        }
        IT_ASSERT(A_dim.size() >= 2 && B_dim.size() >= 2);
        IT_ASSERT(A_dim[A_dim.size() - 1] == B_dim[B_dim.size() - 2]);

        n = B_dim[B_dim.size() - 1];

        Shape result = A_dim;
        result[result.size() - 1] = n;

        return {{result}};
    }

} // namespace infini