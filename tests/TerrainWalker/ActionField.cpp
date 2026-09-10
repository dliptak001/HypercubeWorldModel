#include "ActionField.h"

#include <stdexcept>

void ActionField::Paint(Heading h, int rows, int cols, std::span<float> field)
{
    if (rows < 2 || cols < 2)
        throw std::invalid_argument("ActionField::Paint rows and cols must be >= 2");
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (field.size() != n)
        throw std::invalid_argument("ActionField::Paint field must be rows*cols");
    const int mid_r = rows / 2;
    const int mid_c = cols / 2;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            float v = 0.f;
            switch (h)
            {
            case Heading::North: v = (i < mid_r) ? 1.f : 0.f; break;
            case Heading::South: v = (i >= mid_r) ? 1.f : 0.f; break;
            case Heading::West:  v = (j < mid_c) ? 1.f : 0.f; break;
            case Heading::East:  v = (j >= mid_c) ? 1.f : 0.f; break;
            }
            field[static_cast<size_t>(i * cols + j)] = v;
        }
    }
}
