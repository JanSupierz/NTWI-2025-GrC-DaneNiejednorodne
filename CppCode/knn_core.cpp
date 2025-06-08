#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

using Neighbor = std::tuple<double, int, int>;  // (distance, file_idx, row_idx)

class Data {
public:
    Data(py::array_t<double>& fileData, const std::vector<int>& columns)
        : m_buffer(fileData.request()),
          m_ptr(static_cast<double*>(m_buffer.ptr)),
          m_rows(static_cast<int>(m_buffer.shape[0])),
          m_cols(static_cast<int>(m_buffer.shape[1])),
          m_presentColumns(columns)
    {
        if (m_buffer.ndim != 2) {
            throw std::runtime_error("Expected 2D numpy array");
        }

        // Map from global attribute index → local column index
        for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
            global_to_local[columns[i]] = i;
        }
    }

    double Get(int row, int global_attr_index) const {
        auto it = global_to_local.find(global_attr_index);
        if (it == global_to_local.end()) {
            throw std::out_of_range("Attribute index not present in this file");
        }

        int col = it->second;
        if (row < 0 || row >= m_rows || col < 0 || col >= m_cols) {
            throw std::out_of_range("Row or column index out of bounds");
        }

        return m_ptr[row * m_cols + col];
    }

    int GetNrRows() const { return m_rows; }
    const std::vector<int>& GetColumns() const { return m_presentColumns; }

private:
    py::buffer_info m_buffer;
    double* m_ptr;
    int m_rows;
    int m_cols;
    std::vector<int> m_presentColumns;
    std::unordered_map<int, int> global_to_local;
};

std::vector<std::vector<Neighbor>> knn_for_file(
    py::array_t<double> currentFileData,
    int currentIndex,
    std::vector<py::array_t<double>> allFileData,
    std::vector<std::vector<int>> columnMasks,
    int k,
    int totalAttrs
) {
    const int nrFiles = static_cast<int>(allFileData.size());

    // Pre-load all Data objects
    std::vector<Data> allData;
    allData.reserve(nrFiles);
    for (int i = 0; i < nrFiles; ++i) {
        allData.emplace_back(allFileData[i], columnMasks[i]);
    }

    const Data& currentData = allData[currentIndex];
    const int nrRows = currentData.GetNrRows();

    // Precompute common attributes between file pairs
    std::vector<std::vector<std::vector<int>>> commonAttrs(nrFiles, std::vector<std::vector<int>>(nrFiles));
    for (int i = 0; i < nrFiles; ++i) {
        std::set<int> colSetA(columnMasks[i].begin(), columnMasks[i].end());
        for (int j = 0; j < nrFiles; ++j) {
            if (i == j) continue;
            for (int attr : columnMasks[j]) {
                if (colSetA.count(attr)) {
                    commonAttrs[i][j].push_back(attr);
                }
            }
        }
    }

    std::vector<std::vector<Neighbor>> neighbors_for_file(nrRows);

    // Optional parallelism
    #pragma omp parallel for
    for (int rowIdx = 0; rowIdx < nrRows; ++rowIdx) {
        std::vector<Neighbor> heap;

        for (int j = 0; j < nrFiles; ++j) {
            if (j == currentIndex) continue;

            const Data& otherData = allData[j];
            const std::vector<int>& shared_attrs = commonAttrs[currentIndex][j];
            if (shared_attrs.empty()) continue;

            const int otherRows = otherData.GetNrRows();

            for (int otherRowIdx = 0; otherRowIdx < otherRows; ++otherRowIdx) {
                double dist_sq = 0.0;
                for (int attr : shared_attrs) {
                    double diff = currentData.Get(rowIdx, attr) - otherData.Get(otherRowIdx, attr);
                    dist_sq += diff * diff;
                }

                double dist_rescaled = dist_sq / shared_attrs.size() * totalAttrs;

                if (heap.size() < static_cast<size_t>(k)) {
                    heap.emplace_back(dist_rescaled, j, otherRowIdx);
                    if (heap.size() == static_cast<size_t>(k)) {
                        std::make_heap(heap.begin(), heap.end(), [](const Neighbor& a, const Neighbor& b) {
                            return std::get<0>(a) < std::get<0>(b);  // max-heap
                        });
                    }
                } else if (dist_rescaled < std::get<0>(heap.front())) {
                    std::pop_heap(heap.begin(), heap.end(), [](const Neighbor& a, const Neighbor& b) {
                        return std::get<0>(a) < std::get<0>(b);
                    });
                    heap.back() = { dist_rescaled, j, otherRowIdx };
                    std::push_heap(heap.begin(), heap.end(), [](const Neighbor& a, const Neighbor& b) {
                        return std::get<0>(a) < std::get<0>(b);
                    });
                }
            }
        }

        std::sort(heap.begin(), heap.end(), [](const Neighbor& a, const Neighbor& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

        neighbors_for_file[rowIdx] = std::move(heap);
    }

    return neighbors_for_file;
}

PYBIND11_MODULE(knn_core, m) {
    m.def("knn_for_file", &knn_for_file, "Find k-NN for one file against others");
}