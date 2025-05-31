#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

using Neighbor = std::tuple<double, int, int>;

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
    }

    double Get(int row, int global_attr_index) const
    {
        auto it = std::find(m_presentColumns.begin(), m_presentColumns.end(), global_attr_index);
        if (it == m_presentColumns.end()) {
            throw std::out_of_range("Attribute index not present in this file");
        }

        int col_in_data = static_cast<int>(std::distance(m_presentColumns.begin(), it));

        if (row < 0 || row >= m_rows || col_in_data < 0 || col_in_data >= m_cols) {
            throw std::out_of_range("Row or column index out of bounds");
        }

        return m_ptr[row * m_cols + col_in_data];
    }


    int GetNrRows() const { return m_rows; }
    int GetNrCols() const { return m_cols; }
    const std::vector<int>& GetColumns() const { return m_presentColumns; }

private:
    py::buffer_info m_buffer;
    double* m_ptr;
    int m_rows;
    int m_cols;
    std::vector<int> m_presentColumns;
};

std::vector<std::vector<Neighbor>> knn_for_file(
    py::array_t<double> currentFileData,
    int currentIndex,
    std::vector<py::array_t<double>> allFileData,
    std::vector<std::vector<int>> columnMasks,
    int k,
    int totalAttrs
) {
    int nrFiles = static_cast<int>(allFileData.size());
    Data currentData(currentFileData, columnMasks[currentIndex]);

    std::vector<std::vector<Neighbor>> neighbors_for_file(currentData.GetNrRows());

    for (int rowIdx = 0; rowIdx < currentData.GetNrRows(); ++rowIdx) 
    {
        std::vector<Neighbor> closestNeighbors;
        closestNeighbors.reserve(k);

        for (int j = 0; j < nrFiles; ++j) {
            if (j == currentIndex) continue;

            Data otherData(allFileData[j], columnMasks[j]);

            // Find common attributes
            std::vector<int> common_attrs;

            for (int attr : currentData.GetColumns()) 
            {
                if (std::find(otherData.GetColumns().begin(), otherData.GetColumns().end(), attr) != otherData.GetColumns().end()) 
                {
                    common_attrs.push_back(attr);
                }
            }

            //Eraly quit, no common columns
            if (common_attrs.empty()) continue;


            for (int otherRowIdx = 0; otherRowIdx < otherData.GetNrRows(); ++otherRowIdx) 
            {
                double dist_sq = 0.0;

                for (int attr : common_attrs) 
                {
                    double diff = currentData.Get(rowIdx, attr) - otherData.Get(otherRowIdx, attr);
                    dist_sq += diff * diff;
                }

                double dist_rescaled = dist_sq / common_attrs.size() * totalAttrs;

                if (closestNeighbors.size() < static_cast<size_t>(k)) 
                {
                    closestNeighbors.emplace_back(dist_rescaled, j, otherRowIdx);

                    if (closestNeighbors.size() == static_cast<size_t>(k)) 
                    {
                        std::make_heap(closestNeighbors.begin(), closestNeighbors.end(),
                            [](const Neighbor& a, const Neighbor& b) 
                            {
                                return std::get<0>(a) < std::get<0>(b);  // Max-heap
                            });
                    }
                }
                else if (dist_rescaled < std::get<0>(closestNeighbors.front())) 
                {
                    std::pop_heap(closestNeighbors.begin(), closestNeighbors.end(),
                        [](const Neighbor& a, const Neighbor& b) {
                            return std::get<0>(a) < std::get<0>(b);
                        });

                    closestNeighbors.back() = { dist_rescaled, j, otherRowIdx };
                    std::push_heap(closestNeighbors.begin(), closestNeighbors.end(),
                        [](const Neighbor& a, const Neighbor& b) 
                        {
                            return std::get<0>(a) < std::get<0>(b);
                        });
                }
            }
        }

        std::sort(closestNeighbors.begin(), closestNeighbors.end(),
            [](const Neighbor& a, const Neighbor& b) {
                return std::get<0>(a) < std::get<0>(b);
            });

        neighbors_for_file[rowIdx] = std::move(closestNeighbors);
    }

    return neighbors_for_file;
}

PYBIND11_MODULE(knn_core, m) {
    m.def("knn_for_file", &knn_for_file, "Find k-NN for one file against others");

}
