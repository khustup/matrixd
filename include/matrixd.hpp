#pragma once

#include "impl/matrixd_impl.hpp"

#include <cassert>
#include <span>

namespace khustup {
    namespace impl {
        template <typename T, int abs_size, int abs_offset, int offset, int size, int ... tail>
        struct matrix_impl<T, abs_size, abs_offset, offset, size, tail ...>
        {
            /// @name Properties
            /// @{
            static constexpr inline bool is_consistent = abs_size >= 0 &&
                                                         abs_offset >= 0 &&
                                                         offset >= 0 &&
                                                         size >= 0 &&
                                                         offset + size <= abs_size &&
                                                         matrix_impl<T, tail ...>::is_consistent;
            static_assert(is_consistent);

            static constexpr inline int dimensions = sizeof...(tail) / 4 + 1;

            static constexpr inline int volume = size * matrix_impl<T, tail ...>::volume;

            static constexpr inline int absolute_volume = abs_size * matrix_impl<T, tail ...>::absolute_volume;

            static constexpr inline auto sizes = extracted_sizes<abs_size, abs_offset, offset, size, tail ...>;

            static constexpr inline auto offsets = extracted_offsets<abs_size, abs_offset, offset, size, tail ...>;

            static constexpr inline auto absolute_sizes = extracted_abs_sizes<abs_size,
                                                                              abs_offset,
                                                                              offset,
                                                                              size,
                                                                              tail ...
                                                                             >;

            static constexpr inline auto absolute_offsets = extracted_abs_offsets<abs_size,
                                                                                  abs_offset,
                                                                                  offset,
                                                                                  size,
                                                                                  tail ...
                                                                                 >;

            static constexpr inline bool is_cropped = offset != 0 ||
                                                      size != abs_size ||
                                                      matrix_impl<T, tail ...>::is_cropped;

            static constexpr inline bool axes_swapped = abs_offset != impl::volume<tail ...> ||
                                                        matrix_impl<T, tail ...>::axes_swapped;

            static constexpr inline bool is_continuous = ((!is_cropped) && (!axes_swapped));
            /// @}

            /// @name Utilities
            /// @{
            template <typename H, typename ... I>
            static constexpr inline int raw_offset(H h, I ... i) noexcept
            {
                static_assert(std::is_same<int, H>::value);
                return (h + offset) * abs_offset + matrix_impl<T, tail ...>::raw_offset(i ...);
            }

            static constexpr inline int raw_offset(int h) noexcept
            {
                return (h + offset) * abs_offset;
            }

            template <int i, int j>
            using swap_axes_matrix_type = typename matrix_swap_axes_type<i, j, matrix_impl>::type;

            template <int index_count>
            using submatrix_type = typename impl::submatrix_type<matrix_impl, index_count>::type;

            template <int new_offset, int new_size, int ... new_tail>
            using cropped_matrix_type = typename impl::cropped_matrix_type<matrix_impl<T, tail ...>,
                                                                           matrix_impl<T,
                                                                                       abs_size,
                                                                                       abs_offset,
                                                                                       offset + new_offset,
                                                                                       new_size
                                                                                      >,
                                                                           new_tail ...>::type;

            template <typename M>
            using dot_product_type = typename dot_product_matrix_type_impl<matrix_impl,
                                                                           M,
                                                                           std::integer_sequence<int>>::type;

            template <typename M>
            using max_size_matrix_type = typename max_size_matrix_type_impl<matrix_impl, M, std::integer_sequence<int>>::type;

            using continuous_matrix_type = impl::continuous_matrix_type_from_matrix<matrix_impl>;
            /// @}

            /// @name Construction & Destruction
            /// @{
            constexpr matrix_impl(std::span<T, absolute_volume> d) noexcept
                : data_{d.data()}
                , allocated_{false}
            {
                assert(is_consistent_check());
            }

            constexpr matrix_impl(const matrix_impl& m) noexcept
                : data_{m.data_}
                , allocated_{false}
            {
                if (m.allocated_) {
                    assert(is_continuous);
                    data_ = new T[absolute_volume]{};
                    std::copy(m.data_, m.data_ + absolute_volume, data_);
                    allocated_ = true;
                }
                assert(is_consistent_check());
            }

            template <typename M>
            constexpr matrix_impl(const M& m) noexcept
                : matrix_impl{}
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                static_assert(is_continuous);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[i];
                    }
                }
                assert(is_consistent_check());
            }

            constexpr matrix_impl(matrix_impl&& m) noexcept
                : data_{m.data_}
                , allocated_{m.allocated_}
            {
                m.allocated_ = false;
                assert(is_consistent_check());
                assert(m.is_consistent_check());
            }

            constexpr explicit matrix_impl(std::nullptr_t) noexcept
                : data_{nullptr}
                , allocated_{false}
            {
                static_assert(is_continuous);
                assert(is_consistent_check());
            }

            constexpr explicit matrix_impl() noexcept
                : data_{new T[absolute_volume]{}}
                , allocated_{true}
            {
                static_assert(is_continuous);
                assert(is_consistent_check());
            }

            constexpr explicit matrix_impl(const T& v) noexcept
                : matrix_impl{}
            {
                std::fill(data_, data_ + absolute_volume, v);
                assert(is_consistent_check());
            }

            constexpr matrix_impl(T* s, T* e) noexcept
                : matrix_impl(std::span<T, absolute_volume>{s, e})
            {
                assert(std::distance(s, e) == absolute_volume);
                assert(is_consistent_check());
            }

            ~matrix_impl() noexcept
            {
                if (allocated_) {
                    delete[] data_;
                }
            }

            constexpr matrix_impl& operator=(const matrix_impl& m) noexcept
            {
                if (this != (&m)) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator=(matrix_impl&& m) noexcept
            {
                if (allocated_) {
                    delete[] data_;
                }
                data_ = m.data_;
                allocated_ = m.allocated_;
                m.allocated_ = false;
                assert(is_consistent_check());
                assert(m.is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) = v;
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr continuous_matrix_type copy() const noexcept
            {
                if (allocated_) {
                    return continuous_matrix_type{*this};
                }
                continuous_matrix_type r;
                r = *this;
                return r;
            }
            /// @}

            /// @name Swap axes, Crop, Reshape
            /// @{
            template <int i, int j>
            constexpr swap_axes_matrix_type<i, j> swap_axes() const& noexcept
            {
                return swap_axes_matrix_type<i, j>{data_, data_ + absolute_volume};
            }

            template <int i, int j>
            constexpr swap_axes_matrix_type<i, j> swap_axes() && noexcept
            {
                auto r = swap_axes_matrix_type<i, j>{data_, data_ + absolute_volume};
                r.allocated_ = allocated_;
                allocated_ = false;
                return r;
            }

            template <int new_offset, int new_size, int ... new_tail>
            constexpr cropped_matrix_type<new_offset, new_size, new_tail ...> crop() const& noexcept
            {
                static_assert(sizeof...(new_tail) == sizeof...(tail) / 2);
                return cropped_matrix_type<new_offset, new_size, new_tail ...>{data_, data_ + absolute_volume};
            }

            template <int new_offset, int new_size, int ... new_tail>
            constexpr cropped_matrix_type<new_offset, new_size, new_tail ...> crop() && noexcept
            {
                static_assert(sizeof...(new_tail) == sizeof...(tail) / 2);
                auto r = cropped_matrix_type<new_offset, new_size, new_tail ...>{data_, data_ + absolute_volume};
                r.allocated_ = allocated_;
                allocated_ = false;
                return r;
            }

            template <int ... sizes>
            constexpr continuous_matrixd<T, sizes ...> reshape() const& noexcept
            {
                static_assert(continuous_matrixd<T, sizes ...>::volume == volume);
                static_assert(!is_cropped, "Can't reshape cropped matrix");
                return continuous_matrixd<T, sizes ...>{data_, data_ + volume};
            }

            template <int ... sizes>
            constexpr continuous_matrixd<T, sizes ...> reshape() && noexcept
            {
                static_assert(continuous_matrixd<T, sizes ...>::volume == volume);
                static_assert(!is_cropped, "Can't reshape cropped matrix");
                auto r = continuous_matrixd<T, sizes ...>{data_, data_ + volume};
                r.allocated_ = allocated_;
                allocated_ = false;
                return r;
            }
            /// @}

            /// @name Element Access
            /// @{
            constexpr matrix_impl<T, tail...> operator[](int index) noexcept
            {
                assert(index < size && index >= 0);
                constexpr int vv = matrix_impl<T, tail...>::absolute_volume;
                assert(is_consistent_check());
                return matrix_impl<T, tail...>{data_ + abs_offset * (index + offset),
                                                                data_ + abs_offset * (index + offset) + vv};
            }

            constexpr const matrix_impl<T, tail...> operator[](int index) const noexcept
            {
                assert(index < size && index >= 0);
                constexpr int vv = matrix_impl<T, tail...>::absolute_volume;
                assert(is_consistent_check());
                return matrix_impl<T, tail...>{data_ + abs_offset * (index + offset),
                                                                data_ + abs_offset * (index + offset) + vv};
            }

            template <typename ... I>
            constexpr T& at(I ... indices) noexcept
            {
                static_assert(sizeof...(indices) == sizeof...(tail) / 4 + 1);
                assert(raw_offset(indices ...) < absolute_volume);
                assert(is_consistent_check());
                return data_[raw_offset(indices ...)];
            }

            template <typename ... I>
            constexpr const T& at(I ... indices) const noexcept
            {
                static_assert(sizeof...(indices) == sizeof...(tail) / 4 + 1);
                assert(raw_offset(indices ...) < absolute_volume);
                assert(is_consistent_check());
                return data_[raw_offset(indices ...)];
            }

            template <typename ... I>
            constexpr submatrix_type<sizeof...(I)> sub(I ... indices) noexcept
            {
                static_assert(sizeof...(indices) < sizeof...(tail) / 4 + 1);
                const auto o = raw_offset(indices ...);
                assert(o < absolute_volume);
                constexpr auto v = submatrix_type<sizeof...(indices)>::absolute_volume;
                assert(is_consistent_check());
                return submatrix_type<sizeof...(indices)>{data_ + o, data_ + o + v};
            }

            template <typename ... I>
            constexpr const submatrix_type<sizeof...(I)> sub(I ... indices) const noexcept
            {
                static_assert(sizeof...(indices) < sizeof...(tail) / 4 + 1);
                const auto o = raw_offset(indices ...);
                assert(o < absolute_volume);
                constexpr auto v = submatrix_type<sizeof...(indices)>::absolute_volume;
                assert(is_consistent_check());
                return submatrix_type<sizeof...(indices)>{data_ + o, data_ + o + v};
            }
            /// @}

            /// @name Operations
            /// @{
            template <typename M>
            constexpr matrix_impl& operator+=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) += m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) += m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator+=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) += v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator-=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) -= m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) -= m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator-=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) -= v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator*=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) *= m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) *= m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator*=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) *= v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator/=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) /= m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) /= m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator/=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) /= v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator+(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm += m;
                return mm;
            }

            constexpr auto operator+(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm += v;
                return mm;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator-(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm -= m;
                return mm;
            }

            constexpr auto operator-(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm -= v;
                return mm;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator*(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm *= m;
                return mm;
            }

            constexpr auto operator*(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm *= v;
                return mm;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator/(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm /= m;
                return mm;
            }

            constexpr auto operator/(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm /= v;
                return mm;
            }

            template <typename M>
            constexpr auto dot(const M& m) const noexcept -> dot_product_type<M>
            {
                dot_product_type<M> r;
                constexpr int s0 = std::tuple_size<decltype(sizes)>::value;
                constexpr bool s = s0 == 2;
                constexpr bool s1 = std::get<0>(sizes) == 1;
                constexpr bool s2 = std::get<0>(M::sizes) == 1;
                constexpr bool c = (dot_product_type<M>::volume * std::get<s0 - 1>(sizes)) > crit_compl;
                dot_product_calculator<matrix_impl, M, s, s1, s2, c>::calculate(*this, m, r);
                return r;
            }

            constexpr continuous_matrix_type sqrt() const noexcept
            {
                auto mm = copy();
                impl::sqrt_calculator<continuous_matrix_type>::calculate(mm);
                return mm;
            }
            /// @}

            /// @name Comparison
            /// @{
            constexpr bool operator==(const matrix_impl& m) const noexcept
            {
                if (data_ == m.data_) {
                    return true;
                }
                for (auto i = 0; i < size; ++i) {
                    if (operator[](i) != m[i]) {
                        return false;
                    }
                }
                return true;
            }

            template <typename M>
            constexpr bool operator==(const M& m) const noexcept
            {
                static_assert(sizes == M::sizes);
                for (auto i = 0; i < size; ++i) {
                    if (operator[](i) != m[i]) {
                        return false;
                    }
                }
                return true;
            }

            constexpr bool operator==(std::nullptr_t) const noexcept
            {
                return data_ == nullptr;
            }

            template <typename M>
            constexpr bool operator!=(const M& m) const noexcept
            {
                return !((*this) == m);
            }
            /// @}

            /// @name Access to data.
            /// @{
            T* data() noexcept
            {
                return data_;
            }

            const T* data() const noexcept
            {
                return data_;
            }
            /// @}

            template <typename T1, int ... values>
            friend class matrix_impl;

        private:
            bool is_consistent_check() const noexcept
            {
                return true;
            }

        private:
            T* data_;
            bool allocated_;
        };

        template <typename T, int abs_size, int abs_offset, int offset, int size>
        struct matrix_impl<T, abs_size, abs_offset, offset, size>
        {
            /// @name Properties
            /// @{
            static constexpr inline bool is_consistent = abs_size >= 0 &&
                                                         abs_offset >= 0 &&
                                                         offset >= 0 &&
                                                         size >= 0 &&
                                                         offset + size <= abs_size;
            static_assert(is_consistent);

            static constexpr inline int dimensions = 1;

            static constexpr inline int volume = size;

            static constexpr inline int absolute_volume = abs_size;

            static constexpr inline auto sizes = std::make_tuple(size);

            static constexpr inline auto offsets = std::make_tuple(offset);

            static constexpr inline auto absolute_sizes = std::make_tuple(abs_size);

            static constexpr inline auto absolute_offsets = std::make_tuple(abs_offset);

            static constexpr inline bool is_cropped = (offset != 0 || size != abs_size);

            static constexpr inline bool axes_swapped = (abs_offset != 1);

            static constexpr inline bool is_continuous = ((!is_cropped) && (!axes_swapped));
            /// @}

            /// @name Utilities
            /// @{
            static constexpr inline int raw_offset(int h) noexcept
            {
                return (h + offset) * abs_offset;
            }

            template <int new_offset, int new_size>
            using cropped_matrix_type = matrix_impl<T, abs_size, abs_offset, offset + new_offset, new_size>;

            using continuous_matrix_type = impl::continuous_matrix_type_from_matrix<matrix_impl>;

            template <typename M>
            using max_size_matrix_type = typename max_size_matrix_type_impl<matrix_impl, M, std::integer_sequence<int>>::type;
            /// @}

            /// @name Construction & Destruction
            /// @{
            constexpr matrix_impl(std::span<T, absolute_volume> d) noexcept
                : data_{d.data()}
                , allocated_{false}
            {
                assert(is_consistent_check());
            }

            constexpr matrix_impl(const matrix_impl& m) noexcept
                : data_{m.data_}
                , allocated_{false}
            {
                if (m.allocated_) {
                    assert(is_continuous);
                    data_ = new T[absolute_volume]{};
                    std::copy(m.data_, m.data_ + absolute_volume, data_);
                    allocated_ = true;
                }
                assert(is_consistent_check());
            }


            template <typename M>
            constexpr matrix_impl(const M& m) noexcept
                : matrix_impl{}
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                static_assert(is_continuous);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[i];
                    }
                }
                assert(is_consistent_check());
            }

            constexpr matrix_impl(matrix_impl&& m) noexcept
                : data_{m.data_}
                , allocated_{m.allocated_}
            {
                m.allocated_ = false;
                assert(is_consistent_check());
                assert(m.is_consistent_check());
            }

            constexpr explicit matrix_impl(std::nullptr_t) noexcept
                : data_{nullptr}
                , allocated_{false}
            {
                static_assert(is_continuous);
                assert(is_consistent_check());
            }

            constexpr explicit matrix_impl() noexcept
                : data_{new T[absolute_volume]{}}
                , allocated_{true}
            {
                static_assert(is_continuous);
                assert(is_consistent_check());
            }

            constexpr explicit matrix_impl(const T& v) noexcept
                : matrix_impl{}
            {
                std::fill(data_, data_ + absolute_volume, v);
                assert(is_consistent_check());
            }

            constexpr matrix_impl(T* s, T* e) noexcept
                : matrix_impl(std::span<T, absolute_volume>{s, e})
            {
                assert(std::distance(s, e) == absolute_volume);
                assert(is_consistent_check());
            }

            ~matrix_impl() noexcept
            {
                if (allocated_) {
                    delete[] data_;
                }
            }

            constexpr matrix_impl& operator=(const matrix_impl& m) noexcept
            {
                if (this != (&m)) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator=(matrix_impl&& m) noexcept
            {
                if (allocated_) {
                    delete[] data_;
                }
                data_ = m.data_;
                allocated_ = m.allocated_;
                m.allocated_ = false;
                assert(is_consistent_check());
                assert(m.is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) = v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) = m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr continuous_matrix_type copy() const noexcept
            {
                if (allocated_) {
                    return continuous_matrix_type{*this};
                }
                continuous_matrix_type r;
                r = *this;
                return r;
            }
            /// @}

            /// @name Crop Reshape
            /// @{
            template <int new_offset, int new_size>
            constexpr cropped_matrix_type<new_offset, new_size> crop() const& noexcept
            {
                return cropped_matrix_type<new_offset, new_size>{data_, data_ + absolute_volume};
            }

            template <int new_offset, int new_size, int ... new_tail>
            constexpr cropped_matrix_type<new_offset, new_size> crop() && noexcept
            {
                auto r = cropped_matrix_type<new_offset, new_size>{data_, data_ + absolute_volume};
                r.allocated_ = allocated_;
                allocated_ = false;
                return r;
            }

            template <int ... sizes>
            constexpr continuous_matrixd<T, sizes ...> reshape() const& noexcept
            {
                static_assert(continuous_matrixd<T, sizes ...>::volume == volume);
                static_assert(!is_cropped, "Can't reshape cropped matrix");
                return continuous_matrixd<T, sizes ...>{data_, data_ + volume};
            }

            template <int ... sizes>
            constexpr continuous_matrixd<T, sizes ...> reshape() && noexcept
            {
                static_assert(continuous_matrixd<T, sizes ...>::volume == volume);
                static_assert(!is_cropped, "Can't reshape cropped matrix");
                auto r = continuous_matrixd<T, sizes ...>{data_, data_ + volume};
                r.allocated_ = allocated_;
                allocated_ = false;
                return r;
            }
            /// @}

            /// @name Element Access
            /// @{
            constexpr T& operator[](int index) noexcept
            {
                assert(index < size && index >= 0);
                assert(is_consistent_check());
                return data_[(index + offset) * abs_offset];
            }

            constexpr const T& operator[](int index) const noexcept
            {
                assert(index < size && index >= 0);
                return data_[(index + offset) * abs_offset];
            }

            constexpr T& at(int index) noexcept
            {
                assert(raw_offset(index) < absolute_volume);
                assert(is_consistent_check());
                return data_[(index + offset) * abs_offset];
            }

            constexpr const T& at(int index) const noexcept
            {
                assert(raw_offset(index) < absolute_volume);
                return data_[(index + offset) * abs_offset];
            }
            /// @}

            /// @name Operations
            /// @{
            template <typename M>
            constexpr matrix_impl& operator+=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) += m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) += m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator+=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) += v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator-=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) -= m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) -= m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator-=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) -= v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator*=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) *= m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) *= m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator*=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) *= v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <typename M>
            constexpr matrix_impl& operator/=(const M& m) noexcept
            {
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                if (std::get<0>(M::sizes) == 1) {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) /= m[0];
                    }
                } else {
                    for (auto i = 0; i < size; ++i) {
                        operator[](i) /= m[i];
                    }
                }
                assert(is_consistent_check());
                return *this;
            }

            constexpr matrix_impl& operator/=(const T& v) noexcept
            {
                for (auto i = 0; i < size; ++i) {
                    operator[](i) /= v;
                }
                assert(is_consistent_check());
                return *this;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator+(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm += m;
                return mm;
            }

            constexpr auto operator+(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm += v;
                return mm;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator-(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm -= m;
                return mm;
            }

            constexpr auto operator-(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm -= v;
                return mm;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator*(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm *= m;
                static_assert(std::get<0>(mm.sizes) == 3);
                return mm;
            }

            constexpr auto operator*(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm *= v;
                return mm;
            }

            template <int ... sizes_and_offsets>
            constexpr auto operator/(const matrix_impl<T, sizes_and_offsets ...>& m) const noexcept -> max_size_matrix_type<matrix_impl<T, sizes_and_offsets ...>>
            {
                using M = matrix_impl<T, sizes_and_offsets ...>;
                static_assert(size == std::get<0>(M::sizes) || size == 1 || std::get<0>(M::sizes) == 1);
                max_size_matrix_type<M> mm = copy();
                mm /= m;
                return mm;
            }

            constexpr auto operator/(const T& v) const noexcept -> continuous_matrix_type
            {
                auto mm = copy();
                mm /= v;
                return mm;
            }

            constexpr continuous_matrix_type sqrt() const noexcept
            {
                auto mm = copy();
                impl::sqrt_calculator<continuous_matrix_type>::calculate(mm);
                return mm;
            }
            /// @}

            /// @name Comparison
            /// @{
            constexpr bool operator==(const matrix_impl& m) const noexcept
            {
                if (data_ == m.data_) {
                    return true;
                }
                for (auto i = 0; i < size; ++i) {
                    if (operator[](i) != m[i]) {
                        return false;
                    }
                }
                return true;
            }

            template <typename M>
            constexpr bool operator==(const M& m) const noexcept
            {
                static_assert(sizes == M::sizes);
                for (auto i = 0; i < size; ++i) {
                    if (operator[](i) != m[i]) {
                        return false;
                    }
                }
                return true;
            }

            constexpr bool operator==(std::nullptr_t) const noexcept
            {
                return data_ == nullptr;
            }

            template <typename M>
            constexpr bool operator!=(const M& m) const noexcept
            {
                return !((*this) == m);
            }
            /// @}

            /// @name Access to data.
            /// @{
            T* data() noexcept
            {
                return data_;
            }

            const T* data() const noexcept
            {
                return data_;
            }
            /// @}

            template <typename T1, int ... values>
            friend class matrix_impl;

        private:
            bool is_consistent_check() const noexcept
            {
                return true;
            }

        private:
            T* data_;
            bool allocated_;
        };
    }

    template <typename T, int ... sizes>
    using matrixd = impl::continuous_matrixd<T, sizes ...>;
}
