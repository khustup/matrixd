#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <tuple>

/// @brief Helper Metafunctions.
namespace khustup {

static constexpr int threads = 4;
static constexpr int crit_compl = 1e6;

namespace impl {

template <int ... sizes_and_offsets>
constexpr inline int volume = 1;

template <int abs_size, int abs_offset, int offset, int size, int ... tail>
constexpr inline int volume<abs_size, abs_offset, offset, size, tail ...> = size * volume<tail ...>;

template <int ... sizes_and_offsets>
constexpr inline auto extracted_abs_sizes = std::make_tuple();

template <int abs_size, int abs_offset, int offset, int size, int ... tail>
constexpr inline auto extracted_abs_sizes<abs_size, abs_offset, offset, size, tail ...> =
    std::tuple_cat(std::make_tuple(abs_size), extracted_abs_sizes<tail ...>);

template <int ... sizes_and_offsets>
constexpr inline auto extracted_abs_offsets = std::make_tuple();

template <int abs_size, int abs_offset, int offset, int size, int ... tail>
constexpr inline auto extracted_abs_offsets<abs_size, abs_offset, offset, size, tail ...> =
    std::tuple_cat(std::make_tuple(abs_offset), extracted_abs_offsets<tail ...>);

template <int ... sizes_and_offsets>
constexpr inline auto extracted_sizes = std::make_tuple();

template <int abs_size, int abs_offset, int offset, int size, int ... tail>
constexpr inline auto extracted_sizes<abs_size, abs_offset, offset, size, tail ...> =
    std::tuple_cat(std::make_tuple(size), extracted_sizes<tail ...>);

template <int ... sizes_and_offsets>
constexpr inline auto extracted_offsets = std::make_tuple();

template <int abs_size, int abs_offset, int offset, int size, int ... tail>
constexpr inline auto extracted_offsets<abs_size, abs_offset, offset, size, tail ...> =
    std::tuple_cat(std::make_tuple(offset), extracted_offsets<tail ...>);

template <int ... values>
constexpr inline int product = 1;

template <int value, int ... tail>
constexpr inline int product<value, tail ...> = value * product<tail ...>;

template <int s, typename S>
struct integer_sequence_add_element;

template <int s, int ... values>
struct integer_sequence_add_element<s, std::integer_sequence<int, values ...>>
{
    using type = std::integer_sequence<int, values ..., s>;
};
/*
template <typename S, int ... sizes_and_offsets>
struct extracted_abs_sizes_impl
{
    static constexpr inline auto value = array_from_sequence<S>::value;
};

template <typename S, int abs_size, int abs_offset, int offset, int size, int ... tail>
struct  extracted_abs_sizes_impl<S, abs_size, abs_offset, offset, size, tail ...> :
    public extracted_abs_sizes_impl<typename integer_sequence_add_element<abs_size, S>::type, tail ...>
{
};

template <typename S, int ... sizes_and_offsets>
struct extracted_abs_offsets_impl
{
    static constexpr inline auto value = array_from_sequence<S>::value;
};

template <typename S, int abs_size, int abs_offset, int offset, int size, int ... tail>
struct  extracted_abs_offsets_impl<S, abs_size, abs_offset, offset, size, tail ...> :
    public extracted_abs_offsets_impl<typename integer_sequence_add_element<abs_offset, S>::type, tail ...>
{
};

template <typename S, int ... sizes_and_offsets>
struct extracted_sizes_impl
{
    static constexpr inline auto value = array_from_sequence<S>::value;
};

template <typename S, int abs_size, int abs_offset, int offset, int size, int ... tail>
struct  extracted_sizes_impl<S, abs_size, abs_offset, offset, size, tail ...> :
    public extracted_sizes_impl<typename integer_sequence_add_element<size, S>::type, tail ...>
{
};

template <typename S, int ... sizes_and_offsets>
struct extracted_offsets_impl
{
    static constexpr inline auto value = array_from_sequence<S>::value;
};

template <typename S, int abs_size, int abs_offset, int offset, int size, int ... tail>
struct  extracted_offsets_impl<S, abs_size, abs_offset, offset, size, tail ...> :
    public extracted_offsets_impl<typename integer_sequence_add_element<offset, S>::type, tail ...>
{
};

template <int ... sizes_and_offsets>
constexpr inline auto extracted_abs_sizes = extracted_abs_sizes_impl<std::integer_sequence<int>, sizes_and_offsets ...>::value;

template <int ... sizes_and_offsets>
constexpr inline auto extracted_abs_offsets = extracted_abs_offsets_impl<std::integer_sequence<int>, sizes_and_offsets ...>::value;

template <int ... sizes_and_offsets>
constexpr inline auto extracted_sizes = extracted_sizes_impl<std::integer_sequence<int>, sizes_and_offsets ...>::value;

template <int ... sizes_and_offsets>
constexpr inline auto extracted_offsets = extracted_offsets_impl<std::integer_sequence<int>, sizes_and_offsets ...>::value;
*/

}

}

namespace khustup {

namespace impl {

/// @brief Matrix declaration.
template <typename T, int ... sizes_and_offsets>
struct matrix_impl;

/// @brief Matrix add axis type.
template <int abs_size, int abs_offset, int offset, int size, typename M>
struct matrix_add_0_axis_type;

template <typename T, int abs_size, int abs_offset, int offset, int size, int ... sizes_and_offsets>
struct matrix_add_0_axis_type<abs_size, abs_offset, offset, size, matrix_impl<T, sizes_and_offsets ...>>
{
    using type = matrix_impl<T, sizes_and_offsets ..., abs_size, abs_offset, offset, size>;
};

/// @brief Matrix remove first axis type.
template <typename M>
struct matrix_remove_0_axis_type;

template <typename T, int abs_size, int abs_offset, int offset, int size, int ... tail>
struct matrix_remove_0_axis_type<matrix_impl<T, abs_size, abs_offset, offset, size, tail ...>>
{
    using type = matrix_impl<T, tail ...>;
};

/// @brief Matrix change first axis type.
template <int abs_size0, int abs_offset0, int offset0, int size0, typename M>
struct matrix_set_0_axis_type;

template <int abs_size0, int abs_offset0, int offset0, int size0,
          typename T,
          int abs_size, int abs_offset, int offset, int size,
          int ... tail
         >
struct matrix_set_0_axis_type<abs_size0, abs_offset0, offset0, size0,
                              matrix_impl<T, abs_size, abs_offset, offset, size, tail ...>>
{
    using type = matrix_impl<T, abs_size0, abs_offset0, offset0, size0, tail ...>;
};

/// @brief Submatrix type.
template <typename M, int index_count>
struct submatrix_type :
    public submatrix_type<typename matrix_remove_0_axis_type<M>::type, index_count - 1>
{
};

template <typename M>
struct submatrix_type<M, 0>
{
    using type = M;
};

/// @brief Continuous matrix type.
template <typename T, typename M, int ... sizes>
struct continuous_matrix_type
{
    using type = M;
};

template <typename T, typename M, int abs_size, int ... tail>
struct continuous_matrix_type<T, M, abs_size, tail ...> :
    public continuous_matrix_type<T, typename matrix_add_0_axis_type<abs_size,
                                                                      product<tail ...>,
                                                                      0,
                                                                      abs_size,
                                                                      M>::type, tail ...>
{
};

template <typename T, typename S>
struct continuous_matrix_type_from_sequence;

template <typename T, int ... abs_sizes>
struct continuous_matrix_type_from_sequence<T, std::integer_sequence<int, abs_sizes ...>>
{
    using type = typename continuous_matrix_type<T, matrix_impl<T>, abs_sizes ...>::type;
};

template <typename M, typename S>
struct continuous_matrix_type_from_matrixd;

template <typename T, typename S, int abs_size, int abs_offset, int offset, int size, int ... tail>
struct continuous_matrix_type_from_matrixd<matrix_impl<T, abs_size, abs_offset, offset, size, tail ...>,
                                           S> : public
        continuous_matrix_type_from_matrixd<matrix_impl<T, tail ...>,
                                            typename integer_sequence_add_element<size, S>::type>
{
};

template <typename T, typename S, int abs_size, int abs_offset, int offset, int size>
struct continuous_matrix_type_from_matrixd<matrix_impl<T, abs_size, abs_offset, offset, size>, S>
{
    using type = typename continuous_matrix_type_from_sequence<T,
                                                               typename integer_sequence_add_element<size,
                                                                                            S>::type>::type;
};

template <typename M>
using continuous_matrix_type_from_matrix = typename continuous_matrix_type_from_matrixd<M,
                                                                            std::integer_sequence<int>>::type;

/// @brief Cropped matrix type.
template <typename M1, typename M2, int ... offsets_and_sizes>
struct cropped_matrix_type
{
    using type = M2;
};

template <typename M1, typename M2, int offset, int size, int ... tail>
struct cropped_matrix_type<M1, M2, offset, size, tail ...> :
    public cropped_matrix_type<typename matrix_remove_0_axis_type<M1>::type,
                                    typename matrix_add_0_axis_type<std::get<0>(M1::absolute_sizes),
                                                                    std::get<0>(M1::absolute_offsets),
                                                                    std::get<0>(M1::offsets) + offset,
                                                                    size,
                                                                    M2>::type,
                                    tail ...>
{
};

/// @brief Matrix swap axes type.
template <int i, int abs_size0, int abs_offset0, int offset0, int size0, typename M1, typename M2>
struct matrix_swap_axis_with_0_type_impl;

template <int i,
          int abs_size0,
          int abs_offset0,
          int offset0,
          int size0,
          typename M,
          typename T,
          int abs_size,
          int abs_offset,
          int offset,
          int size,
          int ... tail
         >
struct matrix_swap_axis_with_0_type_impl<i,
                                         abs_size0,
                                         abs_offset0,
                                         offset0,
                                         size0,
                                         M,
                                         matrix_impl<T, abs_size, abs_offset, offset, size, tail ...>> :
    public matrix_swap_axis_with_0_type_impl<i - 1,
                                             abs_size0,
                                             abs_offset0,
                                             offset0,
                                             size0,
                                             typename matrix_add_0_axis_type<abs_size,
                                                                             abs_offset,
                                                                             offset,
                                                                             size,
                                                                             M>::type,
                                             matrix_impl<T, tail ...>
                                            >
{
};

template <int abs_size0,
          int abs_offset0,
          int offset0,
          int size0,
          typename M,
          typename T,
          int abs_size,
          int abs_offset,
          int offset,
          int size,
          int ... tail
         >
struct matrix_swap_axis_with_0_type_impl<0,
                                         abs_size0,
                                         abs_offset0,
                                         offset0,
                                         size0,
                                         M,
                                         matrix_impl<T, abs_size, abs_offset, offset, size, tail ...>> :
    public matrix_swap_axis_with_0_type_impl<sizeof...(tail) / 4,
                                             abs_size,
                                             abs_offset,
                                             offset,
                                             size,
                                             typename matrix_add_0_axis_type<abs_size0,
                                                                             abs_offset0,
                                                                             offset0,
                                                                             size0,
                                                                             M>::type,
                                             matrix_impl<T, tail ...>
                                            >
{
};

template <int abs_size0, int abs_offset0, int offset0, int size0, typename M, typename T>
struct matrix_swap_axis_with_0_type_impl<0, abs_size0, abs_offset0, offset0, size0, M, matrix_impl<T>>
{
    using type = typename matrix_set_0_axis_type<abs_size0, abs_offset0, offset0, size0,  M>::type;
};

template <int i, typename M>
struct matrix_swap_axis_with_0_type;

template <int i, typename T, int abs_size, int abs_offset, int offset, int size, int ... tail>
struct matrix_swap_axis_with_0_type<i, matrix_impl<T, abs_size, abs_offset, offset, size, tail ...>>
{
    static_assert(i >= 0 && i < sizeof...(tail) / 4 + 1);
    using type = typename matrix_swap_axis_with_0_type_impl<i,
                                                            abs_size,
                                                            abs_offset,
                                                            offset,
                                                            size,
                                                            matrix_impl<T>,
                                                            matrix_impl<T,
                                                                        abs_size,
                                                                        abs_offset,
                                                                        offset,
                                                                        size,
                                                                        tail ...>>::type;
};

template <int i, int j, typename M>
struct matrix_swap_axes_type
{
private:
    static constexpr inline int i_ = std::min(i, j);
    static constexpr inline int j_ = std::max(i, j);

public:
    using type = typename matrix_swap_axis_with_0_type<i_,
        typename matrix_swap_axis_with_0_type<j_,
            typename matrix_swap_axis_with_0_type<i_, M>::type>::type>::type;
};

template <int i, typename M>
struct matrix_swap_axes_type<i, i, M>
{
    using type = M;
};

/// @brief Dot product matrix type.
template <typename M1, typename M2, typename S>
struct dot_product_matrix_type_impl :
    public dot_product_matrix_type_impl<typename submatrix_type<M1, 1>::type,
                                        typename submatrix_type<M2, 1>::type,
                                        typename integer_sequence_add_element<std::max(std::get<0>(M1::sizes),
                                                                                       std::get<0>(M2::sizes)),
                                                                              S>::type>
{
    static_assert(std::tuple_size<decltype(M1::sizes)>::value != 2);
    static constexpr inline auto s1 = std::get<0>(M1::sizes);
    static constexpr inline auto s2 = std::get<0>(M2::sizes);
    static_assert(s1 == s2 || s1 == 1 || s2 == 1);
};

template <typename T,
          int abs_size11, int abs_offset11, int offset11, int size11,
          int abs_size12, int abs_offset12, int offset12, int size12,
          int abs_size21, int abs_offset21, int offset21, int size21,
          int abs_size22, int abs_offset22, int offset22, int size22,
          typename S>
struct dot_product_matrix_type_impl<matrix_impl<T,
                                                abs_size11, abs_offset11, offset11, size11,
                                                abs_size12, abs_offset12, offset12, size12>,
                                    matrix_impl<T,
                                                abs_size21, abs_offset21, offset21, size21,
                                                abs_size22, abs_offset22, offset22, size22>,
                                    S>
{
    static_assert(size12 == size21);
    using type = typename continuous_matrix_type_from_sequence<T,
        typename integer_sequence_add_element<size22,
        typename integer_sequence_add_element<size11,
                                              S>::type
                                                  >::type>::type;
};

template <typename M1, typename M2, bool is_square, bool first_is_one, bool second_is_one, bool async>
struct dot_product_calculator
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    static constexpr int ss = std::get<0>(M1::sizes);
    using S1 = typename M1::template submatrix_type<1>;
    using S2 = typename M2::template submatrix_type<1>;
    static constexpr bool s = std::tuple_size<decltype(S1::sizes)>::value == 2;
    static constexpr bool s1 = std::get<0>(S1::sizes) == 1;
    static constexpr bool s2 = std::get<0>(S2::sizes) == 1;

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        if (async && ss >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto i = start; i < end; ++i) {
                        auto rr = r[i];
                        dot_product_calculator<S1, S2, s, s1, s2, false>::calculate(m1[i], m2[i], rr);
                    }
                }, t * ss / threads, (t + 1) * ss / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        }
        else {
            for (auto i = 0; i < ss; ++i) {
                auto rr = r[i];
                dot_product_calculator<S1, S2, s, s1, s2, async>::calculate(m1[i], m2[i], rr);
            }
        }
    }
};

template <typename M1, typename M2, bool is_square, bool first_is_one, bool async>
struct dot_product_calculator<M1, M2, is_square, first_is_one, true, async>
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    static constexpr int ss = std::get<0>(M1::sizes);
    using S1 = typename M1::template submatrix_type<1>;
    using S2 = typename M2::template submatrix_type<1>;
    static constexpr bool s = std::tuple_size<decltype(S1::sizes)>::value == 2;
    static constexpr bool s1 = std::get<0>(S1::sizes) == 1;
    static constexpr bool s2 = std::get<0>(S2::sizes) == 1;

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        if (async && ss >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto i = start; i < end; ++i) {
                        auto rr = r[i];
                        dot_product_calculator<S1, S2, s, s1, s2, false>::calculate(m1[i], m2[0], rr);
                    }
                }, t * ss / threads, (t + 1) * ss / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        } 
        else {
            for (auto i = 0; i < ss; ++i) {
                auto rr = r[i];
                dot_product_calculator<S1, S2, s, s1, s2, async>::calculate(m1[i], m2[0], rr);
            }
        }
    }
};

template <typename M1, typename M2, bool is_square, bool second_is_one, bool async>
struct dot_product_calculator<M1, M2, is_square, true, second_is_one, async>
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    static constexpr int ss = std::get<0>(M2::sizes);
    using S1 = typename M1::template submatrix_type<1>;
    using S2 = typename M2::template submatrix_type<1>;
    static constexpr bool s = std::tuple_size<decltype(S1::sizes)>::value == 2;
    static constexpr bool s1 = std::get<0>(S1::sizes) == 1;
    static constexpr bool s2 = std::get<0>(S2::sizes) == 1;

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        if (async && ss >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto i = start; i < end; ++i) {
                        auto rr = r[i];
                        dot_product_calculator<S1, S2, s, s1, s2, false>::calculate(m1[0], m2[i], rr);
                    }
                }, t * ss / threads, (t + 1) * ss / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        } 
        else {
            for (auto i = 0; i < ss; ++i) {
                auto rr = r[i];
                dot_product_calculator<S1, S2, s, s1, s2, async>::calculate(m1[0], m2[i], rr);
            }
        }
    }
};

template <typename M1, typename M2, bool is_square, bool async>
struct dot_product_calculator<M1, M2, is_square, true, true, async>
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    using S1 = typename M1::template submatrix_type<1>;
    using S2 = typename M2::template submatrix_type<1>;
    static constexpr bool s = std::tuple_size<decltype(S1::sizes)>::value == 2;
    static constexpr bool s1 = std::get<0>(S1::sizes) == 1;
    static constexpr bool s2 = std::get<0>(S2::sizes) == 1;

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        auto rr = r[0];
        dot_product_calculator<S1, S2, s, s1, s2, async>::calculate(m1[0], m2[0], rr);
    }
};

template <typename M1, typename M2, bool async>
struct dot_product_calculator<M1, M2, true, false, false, async>
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    static constexpr int s = std::get<1>(M1::sizes);
    static constexpr int s1 = std::get<0>(M1::sizes);
    static constexpr int s2 = std::get<1>(M2::sizes);
    static_assert(std::get<1>(M1::sizes) == std::get<0>(M2::sizes));

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        if (async && s1 >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto i = start; i < end; ++i) {
                        for (auto j = 0; j < s2; ++j) {
                            for (auto k = 0; k < s; ++k) {
                                r[i][j] += m1[i][k] * m2[k][j];
                            }
                        }
                    }
                }, t * s1 / threads, (t + 1) * s1 / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        }
        else if (async && s2 >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto i = 0; i < s1; ++i) {
                        for (auto j = start; j < end; ++j) {
                            for (auto k = 0; k < s; ++k) {
                                r[i][j] += m1[i][k] * m2[k][j];
                            }
                        }
                    }
                }, t * s2 / threads, (t + 1) * s2 / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        }
        else {
            for (auto i = 0; i < s1; ++i) {
                for (auto j = 0; j < s2; ++j) {
                    for (auto k = 0; k < s; ++k) {
                        r[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }
        }
    }
};

template <typename M1, typename M2, bool async>
struct dot_product_calculator<M1, M2, true, true, false, async>
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    static constexpr int s = std::get<1>(M1::sizes);
    static constexpr int s1 = std::get<0>(M1::sizes);
    static constexpr int s2 = std::get<1>(M2::sizes);
    static_assert(std::get<1>(M1::sizes) == std::get<0>(M2::sizes));

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        if (async && s2 >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto j = start; j < end; ++j) {
                        for (auto k = 0; k < s; ++k) {
                            r[0][j] += m1[0][k] * m2[k][j];
                        }
                    }
                }, t * s2 / threads, (t + 1) * s2 / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        }
        else {
            for (auto j = 0; j < s2; ++j) {
                for (auto k = 0; k < s; ++k) {
                    r[0][j] += m1[0][k] * m2[k][j];
                }
            }
        }
    }
};

template <typename M1, typename M2, bool async>
struct dot_product_calculator<M1, M2, true, false, true, async>
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    static constexpr int s = std::get<1>(M1::sizes);
    static constexpr int s1 = std::get<0>(M1::sizes);
    static constexpr int s2 = std::get<1>(M2::sizes);
    static_assert(std::get<1>(M1::sizes) == std::get<0>(M2::sizes));

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        if (async && s1 >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto i = start; i < end; ++i) {
                        for (auto j = 0; j < s2; ++j) {
                            r[i][j] += m1[i][0] * m2[0][j];
                        }
                    }
                }, t * s1 / threads, (t + 1) * s1 / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        }
        else if (async && s2 >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto i = 0; i < s1; ++i) {
                        for (auto j = start; j < end; ++j) {
                            r[i][j] += m1[i][0] * m2[0][j];
                        }
                    }
                }, t * s2 / threads, (t + 1) * s2 / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        }
        else {
            for (auto i = 0; i < s1; ++i) {
                for (auto j = 0; j < s2; ++j) {
                    r[i][j] += m1[i][0] * m2[0][j];
                }
            }
        }
    }
};

template <typename M1, typename M2, bool async>
struct dot_product_calculator<M1, M2, true, true, true, async>
{
    using type = typename dot_product_matrix_type_impl<M1, M2, std::integer_sequence<int>>::type;
    static constexpr int s = std::get<1>(M1::sizes);
    static constexpr int s1 = std::get<0>(M1::sizes);
    static constexpr int s2 = std::get<1>(M2::sizes);
    static_assert(std::get<1>(M1::sizes) == std::get<0>(M2::sizes));

    inline static void calculate(const M1& m1, const M2& m2, type& r) noexcept
    {
        if (async && s2 >= threads) {
            std::array<std::future<void>, threads> state;
            for (int t = 0; t < threads; ++t) {
                state[t] = std::async(std::launch::async, [&](int start, int end) {
                    for (auto j = start; j < end; ++j) {
                        r[0][j] += m1[0][0] * m2[0][j];
                    }
                }, t * s2 / threads, (t + 1) * s2 / threads);
            }
            for (int t = 0; t < threads; ++t) {
                state[t].get();
            }
        }
        else {
            for (auto j = 0; j < s2; ++j) {
                r[0][j] += m1[0][0] * m2[0][j];
            }
        }
    }
};

/// @brief Max possible size matrix type.
template <typename M1, typename M2, typename S>
struct max_size_matrix_type_impl :
    public max_size_matrix_type_impl<typename submatrix_type<M1, 1>::type,
                                        typename submatrix_type<M2, 1>::type,
                                        typename integer_sequence_add_element<std::max(std::get<0>(M1::sizes),
                                                                                       std::get<0>(M2::sizes)),
                                                                              S>::type>
{
};

template <typename T,
          int abs_size1, int abs_offset1, int offset1, int size1,
          int abs_size2, int abs_offset2, int offset2, int size2,
          typename S>
struct max_size_matrix_type_impl<matrix_impl<T,
                                                abs_size1, abs_offset1, offset1, size1>,
                                    matrix_impl<T,
                                                abs_size2, abs_offset2, offset2, size2>,
                                    S>
{
    using type = typename continuous_matrix_type_from_sequence<T,
        typename integer_sequence_add_element<std::max(size1, size2),
                                              S>::type
                                                  >::type;
};

template <typename M>
struct sqrt_calculator
{
    inline static constexpr void calculate(M& m)
    {
        for (auto i = 0; i < std::get<0>(M::sizes); ++i) {
            auto mm = m[i];
            sqrt_calculator<typename M::template submatrix_type<1>>::calculate(mm);
        }
    }
};

template <typename T, int abs_size, int abs_offset, int offset, int size>
struct sqrt_calculator<matrix_impl<T, abs_size, abs_offset, offset, size>>
{
    inline static constexpr void calculate(matrix_impl<T, abs_size, abs_offset, offset, size>& m)
    {
        for (auto i = 0; i < size; ++i) {
            m[i] = std::sqrt(m[i]);
        }
    }
};

/// @brief Continuous matrix type.
template <typename T, int ... sizes>
using continuous_matrixd = typename continuous_matrix_type<T, matrix_impl<T>, sizes ...>::type;

}

}
