#include <matrixd.hpp>

#include <cmath>
#include <chrono>
#include <iostream>
#include <random>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <gtest/gtest.h>

float data_[] = {
    1.0, 3.0, 4.0, 5.0,
    2.0, 3.0, 5.0, 4.5,
    6.0, 7.0, 8.0, 4.5,
    34.32, 12.11, 1.11, 45.55,
    43.9, 13.1, 80.0, 54.0,
    4.0, 3.0, 2.0, 1.0
};

std::vector<int> data2_(108000);

TEST(matrixd, base_test) {
    static_assert(!khustup::impl::matrix_impl<float, 2, 2, 0, 2, 3, 4, 0, 3, 2, 1, 0, 2>::is_continuous);
    static_assert(khustup::matrixd<float, 4, 3>::volume == 12);
    static_assert(khustup::matrixd<float, 3, 2, 2>::volume == 12);
    static_assert(khustup::matrixd<float, 3, 2, 1, 2>::volume == 12);
    static_assert(khustup::matrixd<float, 4, 3>::is_continuous);
    static_assert(khustup::matrixd<float, 3, 2, 2>::is_continuous);
    static_assert(khustup::matrixd<float, 3, 2, 1, 2>::is_continuous);
    static_assert(khustup::matrixd<float, 4, 3>::dimensions == 2);
    static_assert(khustup::matrixd<float, 3, 2, 2>::dimensions == 3);
    static_assert(khustup::matrixd<float, 3, 2, 1, 2>::dimensions == 4);
    static_assert(khustup::matrixd<float, 4, 3>::sizes == std::make_tuple(4, 3));
    static_assert(khustup::matrixd<float, 3, 2, 2>::sizes == std::make_tuple(3, 2, 2));
    static_assert(khustup::matrixd<float, 3, 2, 1, 2>::sizes == std::make_tuple(3, 2, 1, 2));
    khustup::matrixd<float, 4, 3> mm{data_, data_ + 12};
    auto mm1 = mm.reshape<3, 2, 2>();
    ASSERT_EQ(mm.volume, 12);
    ASSERT_EQ(mm1.volume, 12);
    ASSERT_EQ(mm.sizes, std::make_tuple(4, 3));
    ASSERT_EQ(mm1.sizes, std::make_tuple(3, 2, 2));
    ASSERT_TRUE(mm.is_continuous);
    ASSERT_TRUE(mm1.is_continuous);
    ASSERT_EQ(mm[0][0], 1.0);
    ASSERT_EQ(mm[0][1], 3.0);
    ASSERT_EQ(mm[0][2], 4.0);
    ASSERT_EQ(mm[1][0], 5.0);
    ASSERT_EQ(mm[1][1], 2.0);
    ASSERT_EQ(mm[1][2], 3.0);
    ASSERT_EQ(mm[2][0], 5.0);
    ASSERT_EQ(mm[2][1], 4.5);
    ASSERT_EQ(mm[2][2], 6.0);
    ASSERT_EQ(mm[3][0], 7.0);
    ASSERT_EQ(mm[3][1], 8.0);
    ASSERT_EQ(mm[3][2], 4.5);
    ASSERT_EQ((mm.at(0, 0)), 1.0);
    ASSERT_EQ((mm.at(0, 1)), 3.0);
    ASSERT_EQ((mm.at(0, 2)), 4.0);
    ASSERT_EQ((mm.at(1, 0)), 5.0);
    ASSERT_EQ((mm.at(1, 1)), 2.0);
    ASSERT_EQ((mm.at(1, 2)), 3.0);
    ASSERT_EQ((mm.at(2, 0)), 5.0);
    ASSERT_EQ((mm.at(2, 1)), 4.5);
    ASSERT_EQ((mm.at(2, 2)), 6.0);
    ASSERT_EQ((mm.at(3, 0)), 7.0);
    ASSERT_EQ((mm.at(3, 1)), 8.0);
    ASSERT_EQ((mm.at(3, 2)), 4.5);
    std::generate(data2_.begin(), data2_.end(), std::rand);
    khustup::matrixd<int, 6000, 9, 2> mm3{data2_.data(), data2_.data() + 108000};
    ASSERT_EQ((mm3.at(5999, 8, 1)), mm3[5999][8][1]);
    ASSERT_EQ((mm3.at(5997, 6, 1)), mm3[5997][6][1]);
    ASSERT_EQ((mm3.at(5997, 6, 0)), mm3[5997][6][0]);
    ASSERT_EQ((mm3.at(2977, 6, 0)), mm3[2977][6][0]);
    ASSERT_EQ((mm3.at(2977, 0, 0)), mm3[2977][0][0]);
    ASSERT_EQ((mm3.at(1977, 0, 0)), mm3[1977][0][0]);
    ASSERT_EQ((mm3.at(977, 0, 0)), mm3[977][0][0]);
    ASSERT_EQ((mm3.at(77, 0, 0)), mm3[77][0][0]);
    ASSERT_EQ((mm3.at(7, 0, 0)), mm3[7][0][0]);
    ASSERT_EQ((mm3.at(0, 0, 0)), mm3[0][0][0]);
}

TEST(matrixd, reshape_test) {
    khustup::matrixd<float, 4, 3> mm{data_, data_ + 12};
    auto mm1 = mm.reshape<3, 2, 2>();
    auto mm2 = mm1.reshape<4, 3>();
    ASSERT_EQ(mm2, mm);
    ASSERT_EQ(mm1[0][0][0], mm[0][0]);
    ASSERT_EQ(mm1[0][0][1], mm[0][1]);
    ASSERT_EQ(mm1[0][1][0], mm[0][2]);
    ASSERT_EQ(mm1[0][1][1], mm[1][0]);
    ASSERT_EQ(mm1[1][0][0], mm[1][1]);
    ASSERT_EQ(mm1[1][0][1], mm[1][2]);
    ASSERT_EQ(mm1[1][1][0], mm[2][0]);
    ASSERT_EQ(mm1[1][1][1], mm[2][1]);
    ASSERT_EQ(mm1[2][0][0], mm[2][2]);
    ASSERT_EQ(mm1[2][0][1], mm[3][0]);
    ASSERT_EQ(mm1[2][1][0], mm[3][1]);
    ASSERT_EQ(mm1[2][1][1], mm[3][2]);

    ASSERT_EQ((mm1.at(0, 0, 0)), (mm.at(0, 0)));
    ASSERT_EQ((mm1.at(0, 0, 1)), (mm.at(0, 1)));
    ASSERT_EQ((mm1.at(0, 1, 0)), (mm.at(0, 2)));
    ASSERT_EQ((mm1.at(0, 1, 1)), (mm.at(1, 0)));
    ASSERT_EQ((mm1.at(1, 0, 0)), (mm.at(1, 1)));
    ASSERT_EQ((mm1.at(1, 0, 1)), (mm.at(1, 2)));
    ASSERT_EQ((mm1.at(1, 1, 0)), (mm.at(2, 0)));
    ASSERT_EQ((mm1.at(1, 1, 1)), (mm.at(2, 1)));
    ASSERT_EQ((mm1.at(2, 0, 0)), (mm.at(2, 2)));
    ASSERT_EQ((mm1.at(2, 0, 1)), (mm.at(3, 0)));
    ASSERT_EQ((mm1.at(2, 1, 0)), (mm.at(3, 1)));
    ASSERT_EQ((mm1.at(2, 1, 1)), (mm.at(3, 2)));
}

TEST(matrixd, swap_axes_test) {
    using MM = khustup::impl::matrix_impl<float, 2, 60, 0, 2, 3, 20, 0, 3, 4, 5, 0, 4, 5, 1, 0, 5>;
    static_assert(std::is_same<MM::swap_axes_matrix_type<0, 0>, MM>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<0, 1>, khustup::impl::matrix_impl<float, 3, 20, 0, 3, 2, 60, 0, 2, 4, 5, 0, 4, 5, 1, 0, 5>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<0, 2>, khustup::impl::matrix_impl<float, 4, 5, 0, 4, 3, 20, 0, 3, 2, 60, 0, 2, 5, 1, 0, 5>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<0, 3>, khustup::impl::matrix_impl<float, 5, 1, 0, 5, 3, 20, 0, 3, 4, 5, 0, 4, 2, 60, 0, 2>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<1, 0>, khustup::impl::matrix_impl<float, 3, 20, 0, 3, 2, 60, 0, 2, 4, 5, 0, 4, 5, 1, 0, 5>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<1, 1>, MM>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<1, 2>, khustup::impl::matrix_impl<float, 2, 60, 0, 2, 4, 5, 0, 4, 3, 20, 0, 3, 5, 1, 0, 5>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<1, 3>, khustup::impl::matrix_impl<float, 2, 60, 0, 2, 5, 1, 0, 5, 4, 5, 0, 4, 3, 20, 0, 3>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<2, 0>, khustup::impl::matrix_impl<float, 4, 5, 0, 4, 3, 20, 0, 3, 2, 60, 0, 2, 5, 1, 0, 5>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<2, 1>, khustup::impl::matrix_impl<float, 2, 60, 0, 2, 4, 5, 0, 4, 3, 20, 0, 3, 5, 1, 0, 5>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<2, 2>, MM>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<2, 3>, khustup::impl::matrix_impl<float, 2, 60, 0, 2, 3, 20, 0, 3, 5, 1, 0, 5, 4, 5, 0, 4>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<3, 0>, khustup::impl::matrix_impl<float, 5, 1, 0, 5, 3, 20, 0, 3, 4, 5, 0, 4, 2, 60, 0, 2>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<3, 1>, khustup::impl::matrix_impl<float, 2, 60, 0, 2, 5, 1, 0, 5, 4, 5, 0, 4, 3, 20, 0, 3>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<3, 2>, khustup::impl::matrix_impl<float, 2, 60, 0, 2, 3, 20, 0, 3, 5, 1, 0, 5, 4, 5, 0, 4>>::value);
    static_assert(std::is_same<MM::swap_axes_matrix_type<3, 3>, MM>::value);
    khustup::matrixd<float, 4, 3> mm{data_, data_ + 12};
    auto mm2 = mm.swap_axes<0, 1>();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ(mm[i][j], mm2[j][i]);
        }
    }
    std::generate(data2_.begin(), data2_.end(), std::rand);
    khustup::matrixd<int, 36000, 3> mm3{data2_.data(), data2_.data() + 108000};
    ASSERT_EQ(mm3.sizes, std::make_tuple(36000, 3));
    auto mm4 = mm3.reshape<500, 4, 9, 2, 3>();
    auto mm5 = mm4.swap_axes<1, 2>();
    auto mm6 = mm5.swap_axes<3, 4>();
    auto mm7 = mm5.swap_axes<1, 2>();
    auto mm8 = mm6.swap_axes<0, 2>();
    ASSERT_EQ(mm4, mm7);
    for (auto i1 = 0; i1 < 500; ++i1) {
        for (auto i2 = 0; i2 < 4; ++i2) {
            for (auto i3 = 0; i3 < 9; ++i3) {
                for (auto i4 = 0; i4 < 2; ++i4) {
                    for (auto i5 = 0; i5 < 2; ++i5) {
                        ASSERT_EQ(mm4[i1][i2][i3][i4][i5], mm5[i1][i3][i2][i4][i5]);
                        ASSERT_EQ(mm4[i1][i2][i3][i4][i5], mm6[i1][i3][i2][i5][i4]);
                        ASSERT_EQ(mm5[i1][i3][i2][i4][i5], mm6[i1][i3][i2][i5][i4]);
                        ASSERT_EQ(mm4[i1][i2][i3][i4][i5], mm8[i2][i3][i1][i5][i4]);
                    }
                }
            }
        }
    }
}

TEST(matrixd, copy_assignment_test) {
    std::generate(data2_.begin(), data2_.end(), std::rand);
    khustup::matrixd<int, 36000, 3> mm{data2_.data(), data2_.data() + 108000};
    auto mm1 = mm;
    ASSERT_EQ(mm1, mm);
    auto mm2 = mm;
    ASSERT_EQ(mm2, mm);
    ASSERT_EQ(mm2, mm1);
    mm1 = mm2;
    ASSERT_EQ(mm1, mm2);
    ASSERT_EQ(mm, mm2);
}

TEST(matrixd, crop_matrix_test) {
    using MM = khustup::impl::matrix_impl<float, 2, 60, 0, 2, 3, 20, 0, 3, 4, 5, 0, 4, 5, 1, 0, 5>;
    using M = khustup::matrixd<float, 2, 3, 4, 5>;
    static_assert(std::is_same<M, MM>::value);
    using C1 = M::cropped_matrix_type<1, 1, 0, 3, 2, 2, 1, 2>;
    static_assert(std::is_same<C1,
                               khustup::impl::matrix_impl<float, 2, 60, 1, 1, 3, 20, 0, 3, 4, 5, 2, 2, 5, 1, 1, 2>
                              >::value);
    using C2 = C1::cropped_matrix_type<0, 1, 1, 1, 1, 1, 0, 2>;
    using C3 = M::cropped_matrix_type<1, 1, 1, 1, 3, 1, 1, 2>;
    static_assert(std::is_same<C2,
                               khustup::impl::matrix_impl<float, 2, 60, 1, 1, 3, 20, 1, 1, 4, 5, 3, 1, 5, 1, 1, 2>
                              >::value);
    static_assert(std::is_same<C2, C3>::value);
    using C4 = C3::cropped_matrix_type<-1, 2, -1, 3, -3, 4, -1, 5>;
    static_assert(std::is_same<M, C4>::value);
    using C5 = M::cropped_matrix_type<0, 2, 0, 3, 0, 4, 0, 5>;
    static_assert(std::is_same<M, C5>::value);
    using C6 = M::cropped_matrix_type<0, 1, 0, 1, 0, 2, 0, 2>;
    static_assert(std::is_same<C6,
                               khustup::impl::matrix_impl<float, 2, 60, 0, 1, 3, 20, 0, 1, 4, 5, 0, 2, 5, 1, 0, 2>
                              >::value);
    std::generate(data2_.begin(), data2_.end(), std::rand);
    {
        khustup::matrixd<int, 108000> mm{data2_.data(), data2_.data() + 108000};
        auto mm1 = mm.crop<40000, 1000>();
        for (auto i1 = 0; i1 < 1000; ++i1) {
            ASSERT_EQ(mm1[i1], mm[40000 + i1]);
        }
    }
    {
        khustup::matrixd<int, 108, 1000> mm{data2_.data(), data2_.data() + 108000};
        auto mm1 = mm.crop<54, 54, 40, 132>();
        for (auto i1 = 0; i1 < 54; ++i1) {
            for (auto i2 = 0; i2 < 132; ++i2) {
                ASSERT_EQ(mm1[i1][i2], mm[54 + i1][40 + i2]);
            }
        }
    }
    {
        khustup::matrixd<int, 10, 60, 60, 3> mm{data2_.data(), data2_.data() + 108000};
        constexpr auto o1 = 0;
        constexpr auto s1 = 5;
        constexpr auto o2 = 10;
        constexpr auto s2 = 50;
        constexpr auto o3 = 20;
        constexpr auto s3 = 20;
        constexpr auto o4 = 1;
        constexpr auto s4 = 1;
        auto mm1 = mm.crop<o1, s1, o2, s2, o3, s3, o4, s4>();
        int x = 0;
        static_assert(mm.raw_offset(0, 10, 20, 1) == mm1.raw_offset(0, 0, 0, 0));
        static_assert(mm.raw_offset(3, 12, 22, 1) == mm1.raw_offset(3, 2, 2, 0));
        static_assert(mm.raw_offset(4, 59, 39, 1) == mm1.raw_offset(4, 49, 19, 0));
        static_assert(mm.raw_offset(4, 59, 39, 1) == mm1.raw_offset(4, 49, 19, 0));
        for (auto i1 = 0; i1 < s1; ++i1) {
            for (auto i2 = 0; i2 < s2; ++i2) {
                for (auto i3 = 0; i3 < s3; ++i3) {
                    for (auto i4 = 0; i4 < s4; ++i4) {
                        ASSERT_EQ(mm1[i1][i2][i3][i4], mm[o1 + i1][o2 + i2][o3 + i3][o4 + i4]);
                    }
                }
            }
        }
    }
}

TEST(matrixd, sub_matrix_test) {
    using M = khustup::matrixd<float, 20, 31, 42, 53>;
    using C1 = M::submatrix_type<1>;
    static_assert(std::is_same<C1,
                               khustup::matrixd<float, 31, 42, 53>
                              >::value);
    using C2 = C1::submatrix_type<1>;
    using C3 = M::submatrix_type<2>;
    static_assert(std::is_same<C2,
                               khustup::matrixd<float, 42, 53>
                              >::value);
    static_assert(std::is_same<C2, C3>::value);
    using C4 = M::submatrix_type<3>;
    using C5 = C1::submatrix_type<2>;
    using C6 = C2::submatrix_type<1>;
    static_assert(std::is_same<C4,
                               khustup::matrixd<float, 53>
                              >::value);
    static_assert(std::is_same<C4, C5>::value);
    static_assert(std::is_same<C4, C6>::value);
    std::generate(data2_.begin(), data2_.end(), std::rand);
    {
        khustup::matrixd<int, 108, 1000> mm{data2_.data(), data2_.data() + 108000};
        auto mm1 = mm.sub(0);
        for (auto i2 = 0; i2 < 1000; ++i2) {
            ASSERT_EQ(mm1[i2], mm[0][i2]);
        }
        mm1 = mm.sub(30);
        for (auto i2 = 0; i2 < 1000; ++i2) {
            ASSERT_EQ(mm1[i2], mm[30][i2]);
        }
        mm1 = mm.sub(107);
        for (auto i2 = 0; i2 < 1000; ++i2) {
            ASSERT_EQ(mm1[i2], mm[107][i2]);
        }
    }
    {
        khustup::matrixd<int, 5, 120, 60, 3> mm{data2_.data(), data2_.data() + 108000};
        auto mm1 = mm.sub(0);
        for (auto i2 = 0; i2 < 120; ++i2) {
            for (auto i3 = 0; i3 < 60; ++i3) {
                for (auto i4 = 0; i4 < 3; ++i4) {
                    ASSERT_EQ(mm1[i2][i3][i4], mm[0][i2][i3][i4]);
                }
            }
        }
        mm1 = mm.sub(3);
        for (auto i2 = 0; i2 < 120; ++i2) {
            for (auto i3 = 0; i3 < 60; ++i3) {
                for (auto i4 = 0; i4 < 3; ++i4) {
                    ASSERT_EQ(mm1[i2][i3][i4], mm[3][i2][i3][i4]);
                }
            }
        }
        mm1 = mm.sub(4);
        for (auto i2 = 0; i2 < 120; ++i2) {
            for (auto i3 = 0; i3 < 60; ++i3) {
                for (auto i4 = 0; i4 < 3; ++i4) {
                    ASSERT_EQ(mm1[i2][i3][i4], mm[4][i2][i3][i4]);
                }
            }
        }
        auto mm2 = mm.sub(0, 5);
        for (auto i3 = 0; i3 < 60; ++i3) {
            for (auto i4 = 0; i4 < 3; ++i4) {
                ASSERT_EQ(mm2[i3][i4], mm[0][5][i3][i4]);
            }
        }
        mm2 = mm.sub(3, 43);
        for (auto i3 = 0; i3 < 60; ++i3) {
            for (auto i4 = 0; i4 < 3; ++i4) {
                ASSERT_EQ(mm2[i3][i4], mm[3][43][i3][i4]);
            }
        }
        mm2 = mm.sub(4, 100);
        for (auto i3 = 0; i3 < 60; ++i3) {
            for (auto i4 = 0; i4 < 3; ++i4) {
                ASSERT_EQ(mm2[i3][i4], mm[4][100][i3][i4]);
            }
        }
        auto mm3 = mm.sub(0, 5, 7);
        for (auto i4 = 0; i4 < 3; ++i4) {
            ASSERT_EQ(mm3[i4], mm[0][5][7][i4]);
        }
        mm3 = mm.sub(3, 43, 41);
        for (auto i4 = 0; i4 < 3; ++i4) {
            ASSERT_EQ(mm3[i4], mm[3][43][41][i4]);
        }
        mm3 = mm.sub(4, 100, 59);
        for (auto i4 = 0; i4 < 3; ++i4) {
            ASSERT_EQ(mm3[i4], mm[4][100][59][i4]);
        }
    }
}

TEST(matrixd, crop_swap_axes_matrix_test) {
    using M = khustup::matrixd<float, 2, 3, 4, 5>;
    using S1 = M::swap_axes_matrix_type<1, 3>;
    using C1 = S1::cropped_matrix_type<1, 1, 0, 3, 2, 2, 1, 2>;
    static_assert(std::is_same<C1,
                               khustup::impl::matrix_impl<float, 2, 60, 1, 1, 5, 1, 0, 3, 4, 5, 2, 2, 3, 20, 1, 2>
                              >::value);
    using C2 = M::cropped_matrix_type<1, 1, 0, 3, 2, 2, 1, 2>;
    using S2 = C2::swap_axes_matrix_type<1, 3>;
    static_assert(std::is_same<S2,
                               khustup::impl::matrix_impl<float, 2, 60, 1, 1, 5, 1, 1, 2, 4, 5, 2, 2, 3, 20, 0, 3>
                              >::value);
    std::generate(data2_.begin(), data2_.end(), std::rand);
    {
        khustup::matrixd<int, 108, 1000> mm{data2_.data(), data2_.data() + 108000};
        auto mm1 = mm.crop<49, 54, 41, 501>();
        auto mm2 = mm1.swap_axes<0, 1>();
        for (auto i1 = 0; i1 < 54; ++i1) {
            for (auto i2 = 0; i2 < 501; ++i2) {
                ASSERT_EQ(mm1[i1][i2], mm[49 + i1][41 + i2]);
                ASSERT_EQ(mm2[i2][i1], mm[49 + i1][41 + i2]);
            }
        }
        auto mm3 = mm.swap_axes<0, 1>();
        auto mm4 = mm3.crop<141, 581, 39, 54>();
        for (auto i1 = 0; i1 < 581; ++i1) {
            for (auto i2 = 0; i2 < 54; ++i2) {
                ASSERT_EQ(mm4[i1][i2], mm[39 + i2][141 + i1]);
            }
        }
    }
    {
        khustup::matrixd<int, 10, 60, 60, 3> mm{data2_.data(), data2_.data() + 108000};
        constexpr auto o1 = 2;
        constexpr auto s1 = 6;
        constexpr auto o2 = 11;
        constexpr auto s2 = 38;
        constexpr auto o3 = 28;
        constexpr auto s3 = 28;
        constexpr auto o4 = 0;
        constexpr auto s4 = 1;
        auto mm1 = mm.crop<o1, s1, o2, s2, o3, s3, o4, s4>();
        auto mm2 = mm1.swap_axes<1, 2>();
        for (auto i1 = 0; i1 < s1; ++i1) {
            for (auto i2 = 0; i2 < s2; ++i2) {
                for (auto i3 = 0; i3 < s3; ++i3) {
                    for (auto i4 = 0; i4 < s4; ++i4) {
                        ASSERT_EQ(mm2[i1][i3][i2][i4], mm[o1 + i1][o2 + i2][o3 + i3][o4 + i4]);
                    }
                }
            }
        }
        auto mm3 = mm.swap_axes<1, 3>();
        auto mm4 = mm3.crop<o1, s1, o4, s4, o3, s3, o2, s2>();
        for (auto i1 = 0; i1 < s1; ++i1) {
            for (auto i2 = 0; i2 < s2; ++i2) {
                for (auto i3 = 0; i3 < s3; ++i3) {
                    for (auto i4 = 0; i4 < s4; ++i4) {
                        ASSERT_EQ(mm4[i1][i4][i3][i2], mm[o1 + i1][o2 + i2][o3 + i3][o4 + i4]);
                    }
                }
            }
        }
        auto mm5 = mm4.swap_axes<2, 3>();
        for (auto i1 = 0; i1 < s1; ++i1) {
            for (auto i2 = 0; i2 < s2; ++i2) {
                for (auto i3 = 0; i3 < s3; ++i3) {
                    for (auto i4 = 0; i4 < s4; ++i4) {
                        ASSERT_EQ(mm5[i1][i4][i2][i3], mm[o1 + i1][o2 + i2][o3 + i3][o4 + i4]);
                    }
                }
            }
        }
    }
}

TEST(matrixd, element_wise_operations_test) {
    std::vector<int> d1(108000);
    std::vector<int> d2(108000);
    std::vector<int> d3(108000);
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        std::generate(d2.begin(), d2.end(), []() {
                static int o = 0;
                return ++o / 2;
            });
        std::generate(d3.begin(), d3.end(), []() {
                static int o = 0;
                ++o;
                return o + o / 2;
            });
        khustup::matrixd<int, 108000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108000> mm3{d3.data(), d3.data() + 108000};
        mm1 += mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        khustup::matrixd<int, 108, 1000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm3{d3.data(), d3.data() + 108000};
        mm1 += mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        khustup::matrixd<int, 27, 4, 250, 4> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm3{d3.data(), d3.data() + 108000};
        mm1 += mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        std::generate(d2.begin(), d2.end(), []() {
                static int o = 0;
                return ++o / 2;
            });
        std::generate(d3.begin(), d3.end(), []() {
                static int o = 0;
                ++o;
                return o - o / 2;
            });
        khustup::matrixd<int, 108000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108000> mm3{d3.data(), d3.data() + 108000};
        mm1 -= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        khustup::matrixd<int, 108, 1000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm3{d3.data(), d3.data() + 108000};
        mm1 -= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        khustup::matrixd<int, 27, 4, 250, 4> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm3{d3.data(), d3.data() + 108000};
        mm1 -= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                ++o;
                return o;
            });
        std::generate(d2.begin(), d2.end(), []() {
                static int o = 0;
                ++o;
                return o / 2;
            });
        std::generate(d3.begin(), d3.end(), []() {
                static int o = 0;
                ++o;
                return o * (o / 2);
            });
        khustup::matrixd<int, 108000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108000> mm3{d3.data(), d3.data() + 108000};
        mm1 *= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                ++o;
                return o;
            });
        khustup::matrixd<int, 108, 1000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm3{d3.data(), d3.data() + 108000};
        mm1 *= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                ++o;
                return o;
            });
        khustup::matrixd<int, 27, 4, 250, 4> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm3{d3.data(), d3.data() + 108000};
        mm1 *= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 1;
                ++o;
                return o * o;
            });
        std::generate(d2.begin(), d2.end(), []() {
                static int o = 1;
                ++o;
                return o / 2;
            });
        std::generate(d3.begin(), d3.end(), []() {
                static int o = 1;
                ++o;
                return o * o / (o / 2);
            });
        khustup::matrixd<int, 108000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108000> mm3{d3.data(), d3.data() + 108000};
        mm1 /= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 1;
                ++o;
                return o * o;
            });
        khustup::matrixd<int, 108, 1000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm3{d3.data(), d3.data() + 108000};
        mm1 /= mm2;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 1;
                ++o;
                return o * o;
            });
        khustup::matrixd<int, 27, 4, 250, 4> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm2{d2.data(), d2.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm3{d3.data(), d3.data() + 108000};
        mm1 /= mm2;
        ASSERT_EQ(mm1, mm3);
    }
}

TEST(matrixd, operations_with_numbers_test) {
    std::vector<int> d1(108000);
    std::vector<int> d3(108000);
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        std::generate(d3.begin(), d3.end(), []() {
                static int o = 5;
                ++o;
                return o;
            });
        khustup::matrixd<int, 108000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108000> mm3{d3.data(), d3.data() + 108000};
        mm1 += 5;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 11;
                ++o;
                return o;
            });
        khustup::matrixd<int, 108, 1000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108, 1000> mm3{d3.data(), d3.data() + 108000};
        mm1 -= 6;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                return ++o;
            });
        std::generate(d3.begin(), d3.end(), []() {
                static int o = 0;
                ++o;
                return 5 * o;
            });
        khustup::matrixd<int, 27, 4, 250, 4> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 27, 4, 250, 4> mm3{d3.data(), d3.data() + 108000};
        mm1 *= 5;
        ASSERT_EQ(mm1, mm3);
    }
    {
        std::generate(d1.begin(), d1.end(), []() {
                static int o = 0;
                ++o;
                return 12 * o;
            });
        std::generate(d3.begin(), d3.end(), []() {
                static int o = 0;
                ++o;
                return o;
            });
        khustup::matrixd<int, 108000> mm1{d1.data(), d1.data() + 108000};
        khustup::matrixd<int, 108000> mm3{d3.data(), d3.data() + 108000};
        mm1 /= 12;
        ASSERT_EQ(mm1, mm3);
    }
}

TEST(matrixd, operations_copy_test) {
    {
        khustup::matrixd<double, 2, 2, 2> m1{4.3};
        khustup::matrixd<double, 4, 2> m2{3.7};
        auto m3 = m1.crop<1, 1, 1, 1, 1, 1>() + 1.0;
        ASSERT_EQ(m3.at(0, 0, 0), 5.3);
        auto m4 = m1 + m2.reshape<2, 2, 2>();
        ASSERT_EQ(m4.at(0, 0, 0), 8.0);
        ASSERT_EQ(m4.at(0, 0, 1), 8.0);
        ASSERT_EQ(m4.at(0, 1, 0), 8.0);
        ASSERT_EQ(m4.at(0, 1, 1), 8.0);
        ASSERT_EQ(m4.at(1, 0, 0), 8.0);
        ASSERT_EQ(m4.at(1, 0, 1), 8.0);
        ASSERT_EQ(m4.at(1, 1, 0), 8.0);
        ASSERT_EQ(m4.at(1, 1, 1), 8.0);
        m4 = m4 - m1;
        ASSERT_EQ(m4, (m2.reshape<2, 2, 2>()));
        m1[0] = 2.0;
        m1[1] = 3.0;
        m4 = m1.swap_axes<0, 2>() * m2.reshape<2, 2, 2>();
        ASSERT_TRUE(std::abs(m4.at(0, 0, 0) - 7.4) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(0, 0, 1) - 11.1) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(0, 1, 0) - 7.4) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(0, 1, 1) - 11.1) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 0, 0) - 7.4) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 0, 1) - 11.1) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 1, 0) - 7.4) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 1, 1) - 11.1) < 0.0001);
        m1[0] = 2.0;
        m1[1] = 3.0;
        m2 = 15.0;
        m1[0] = 3.0;
        m1[1] = 2.0;
        m4 = m2.reshape<2, 2, 2>() / m1.swap_axes<0, 1>();
        ASSERT_TRUE(std::abs(m4.at(0, 0, 0) - 5.0) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(0, 0, 1) - 5.0) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(0, 1, 0) - 7.5) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(0, 1, 1) - 7.5) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 0, 0) - 5.0) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 0, 1) - 5.0) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 1, 0) - 7.5) < 0.0001);
        ASSERT_TRUE(std::abs(m4.at(1, 1, 1) - 7.5) < 0.0001);
    }
}

TEST(matrixd, dot_product_test) {
    static_assert(std::is_same<khustup::matrixd<float, 4, 5>::dot_product_type<khustup::matrixd<float, 5, 3>>,
                               khustup::matrixd<float, 4, 3>>::value);
    static_assert(std::is_same<khustup::matrixd<float, 10, 4, 50>::dot_product_type<khustup::matrixd<float, 10,  50, 3>>,
                               khustup::matrixd<float, 10, 4, 3>>::value);
    static_assert(std::is_same<khustup::matrixd<float, 1, 4, 50>::dot_product_type<khustup::matrixd<float, 10,  50, 3>>,
                               khustup::matrixd<float, 10, 4, 3>>::value);
    static_assert(std::is_same<khustup::matrixd<float, 300, 10, 4, 50>::dot_product_type<khustup::matrixd<float, 300, 10,  50, 3>>,
                               khustup::matrixd<float, 300, 10, 4, 3>>::value);
    static_assert(std::is_same<khustup::matrixd<float, 231, 1, 4, 50>::dot_product_type<khustup::matrixd<float, 231, 10,  50, 3>>,
                               khustup::matrixd<float, 231, 10, 4, 3>>::value);
    {
        khustup::matrixd<int, 1, 1> m1{3};
        khustup::matrixd<int, 1, 3> m2{2};
        auto m = m1.dot(m2);
        for (auto i = 0; i < 1; ++i) {
            for (auto j = 0; j < 3; ++j) {
                ASSERT_EQ(m[i][j], 6);
            }
        }
    }
    {
        khustup::matrixd<int, 1, 5> m1{3};
        khustup::matrixd<int, 5, 3> m2{2};
        auto m = m1.dot(m2);
        for (auto i = 0; i < 1; ++i) {
            for (auto j = 0; j < 3; ++j) {
                ASSERT_EQ(m[i][j], 30);
            }
        }
    }
    {
        khustup::matrixd<int, 2, 1> m1{3};
        khustup::matrixd<int, 1, 3> m2{2};
        auto m = m1.dot(m2);
        for (auto i = 0; i < 2; ++i) {
            for (auto j = 0; j < 3; ++j) {
                ASSERT_EQ(m[i][j], 6);
            }
        }
    }
    {
        khustup::matrixd<int, 2, 2> m1{3};
        khustup::matrixd<int, 2, 3> m2{2};
        auto m = m1.dot(m2);
        for (auto i = 0; i < 2; ++i) {
            for (auto j = 0; j < 3; ++j) {
                ASSERT_EQ(m[i][j], 12);
            }
        }
    }
    {
        khustup::matrixd<int, 100, 5, 3> m1{3};
        khustup::matrixd<int, 100, 3, 4> m2{5};
        auto m = m1.dot(m2);
        for (auto i = 0; i < 100; ++i) {
            for (auto j = 0; j < 5; ++j) {
                for (auto k = 0; k < 4; ++k) {
                    ASSERT_EQ(m[i][j][k], 45);
                }
            }
        }
    }
    {
        khustup::matrixd<int, 10, 100, 5, 9> m1{3};
        khustup::matrixd<int, 10, 100, 9, 4> m2{40};
        auto m = m1.dot(m2);
        for (auto i = 0; i < 10; ++i) {
            for (auto j = 0; j < 100; ++j) {
                for (auto k = 0; k < 5; ++k) {
                    for (auto l = 0; l < 4; ++l) {
                        ASSERT_EQ(m[i][j][k][l], 1080);
                    }
                }
            }
        }
    }
    {
        khustup::matrixd<float, 36, 10, 100, 5, 90> m1{3.3f};
        khustup::matrixd<float, 100, 10, 36, 90, 4> m2{4.31f};
        auto m3 = m2.swap_axes<0, 2>();
        auto m = m1.dot(m3);
        for (auto i = 0; i < 36; ++i) {
            for (auto j = 0; j < 10; ++j) {
                for (auto k = 0; k < 100; ++k) {
                    for (auto l = 0; l < 5; ++l) {
                        for (auto n = 0; n < 4; ++n) {
                            ASSERT_TRUE(std::abs(m[i][j][k][l][n] - 1280.07f) < 0.01f);
                        }
                    }
                }
            }
        }
    }
    {
        khustup::matrixd<float, 36, 10, 100, 5, 90> m1{3.3f};
        khustup::matrixd<float, 72, 300, 30, 100, 8> m2{4.31f};
        auto m3 = m2.swap_axes<1, 2>().crop<4, 36, 20, 10, 122, 100, 5, 90, 3, 4>();
        auto m = m1.dot(m3);
        for (auto i = 0; i < 36; ++i) {
            for (auto j = 0; j < 10; ++j) {
                for (auto k = 0; k < 100; ++k) {
                    for (auto l = 0; l < 5; ++l) {
                        for (auto n = 0; n < 4; ++n) {
                            ASSERT_TRUE(std::abs(m[i][j][k][l][n] - 1280.07f) < 0.01f);
                        }
                    }
                }
            }
        }
    }
    {
        khustup::matrixd<float, 36, 10, 100, 5, 90> m1{3.3f};
        khustup::matrixd<float, 360000, 36> m2{4.31f};
        auto m3 = m2.reshape<36, 10, 100, 90, 4>();
        auto m = m1.dot(m3);
        for (auto i = 0; i < 36; ++i) {
            for (auto j = 0; j < 10; ++j) {
                for (auto k = 0; k < 100; ++k) {
                    for (auto l = 0; l < 5; ++l) {
                        for (auto n = 0; n < 4; ++n) {
                            ASSERT_TRUE(std::abs(m[i][j][k][l][n] - 1280.07f) < 0.01f);
                        }
                    }
                }
            }
        }
    }
    {
        khustup::matrixd<float, 18, 36000> m1{};
        khustup::matrixd<float, 36000, 36> m2{};
        std::random_device device;
        std::mt19937 generator{device()};
        std::uniform_real_distribution<> distribution{0.0, 2.0};
        std::generate(m1.data(), m1.data() + 18 * 36000, [&distribution, &generator]() {
                return distribution(generator);
            });
        std::generate(m2.data(), m2.data() + 36 * 36000, [&distribution, &generator]() {
                return distribution(generator);
            });
        auto start = std::chrono::high_resolution_clock::now();
        auto m = m1.dot(m2);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = end - start;
        std::cout << "Big sum 2D MATRIXD: " << d.count() << std::endl;
        khustup::matrixd<float, 18, 36> mb{};
        start = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 18, 36, 36000, 1.0, m1.data(), 36000, m2.data(), 36, 0.0, mb.data(), 36);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> db = end - start;
        std::cout << "Big sum 2D BLAS: " << db.count() << std::endl;
        std::cout << "BLAS faster: x" << d.count() / db.count() << std::endl << std::endl;
        for (auto i = 0; i < 18; ++i) {
            for (auto j = 0; j < 36; ++j) {
                ASSERT_TRUE(std::abs(m[i][j] - mb[i][j]) / m[i][j] < 0.00001);
            }
        }
    }
    {
        khustup::matrixd<float, 18, 35> m1{};
        khustup::matrixd<float, 35, 36000> m2{};
        std::random_device device;
        std::mt19937 generator{device()};
        std::uniform_real_distribution<> distribution{0.0, 2.0};
        std::generate(m1.data(), m1.data() + 18 * 35, [&distribution, &generator]() {
                return distribution(generator);
            });
        std::generate(m2.data(), m2.data() + 36000 * 35, [&distribution, &generator]() {
                return distribution(generator);
            });
        auto start = std::chrono::high_resolution_clock::now();
        auto m = m1.dot(m2);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = end - start;
        std::cout << "Small sum 2D MATRIXD: " << d.count() << std::endl;
        khustup::matrixd<float, 18, 36000> mb{};
        start = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 18, 36000, 35, 1.0, m1.data(), 35, m2.data(), 36000, 0.0, mb.data(), 36000);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> db = end - start;
        std::cout << "Small sum 2D BLAS: " << db.count() << std::endl;
        std::cout << "BLAS faster: x" << d.count() / db.count() << std::endl << std::endl;
        for (auto i = 0; i < 18; ++i) {
            for (auto j = 0; j < 36000; ++j) {
                ASSERT_TRUE(std::abs(m[i][j] - mb[i][j]) / m[i][j] < 0.00001);
            }
        }
    }
    {
        khustup::matrixd<float, 20, 18, 35> m1{};
        khustup::matrixd<float, 20, 35, 10000> m2{};
        std::random_device device;
        std::mt19937 generator{device()};
        std::uniform_real_distribution<> distribution{0.0, 2.0};
        std::generate(m1.data(), m1.data() + 18 * 35, [&distribution, &generator]() {
                return distribution(generator);
            });
        std::generate(m2.data(), m2.data() + 10000 * 35, [&distribution, &generator]() {
                return distribution(generator);
            });
        auto start = std::chrono::high_resolution_clock::now();
        auto m = m1.dot(m2);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = end - start;
        std::cout << "Small sum 3D MATRIXD: " << d.count() << std::endl;
        khustup::matrixd<float, 20, 18, 10000> mb{};
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; ++i) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 18, 10000, 35, 1.0, m1.data() + i * 18 * 35, 35, m2.data() + i * 35 * 10000, 10000, 0.0, mb.data() + i * 18 * 10000, 10000);
        }
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> db = end - start;
        std::cout << "Small sum 3D BLAS: " << db.count() << std::endl;
        std::cout << "BLAS faster: x" << d.count() / db.count() << std::endl << std::endl;
    }
    {
        khustup::matrixd<float, 5, 20, 18, 35> m1{};
        khustup::matrixd<float, 5, 20, 35, 10000> m2{};
        std::random_device device;
        std::mt19937 generator{device()};
        std::uniform_real_distribution<> distribution{0.0, 2.0};
        std::generate(m1.data(), m1.data() + 18 * 35, [&distribution, &generator]() {
                return distribution(generator);
            });
        std::generate(m2.data(), m2.data() + 10000 * 35, [&distribution, &generator]() {
                return distribution(generator);
            });
        auto start = std::chrono::high_resolution_clock::now();
        auto m = m1.dot(m2);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = end - start;
        std::cout << "Small sum 4D MATRIXD: " << d.count() << std::endl;
        khustup::matrixd<float, 5, 20, 18, 10000> mb{};
        start = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < 5; ++k) {
            for (int i = 0; i < 20; ++i) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 18, 10000, 35, 1.0, m1.data() + (k * 20 + i) * 18 * 35, 35, m2.data() + (k * 20 + i) * 35 * 10000, 10000, 0.0, mb.data() + (k * 20 + i) * 18 * 10000, 10000);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> db = end - start;
        std::cout << "Small sum 4D BLAS: " << db.count() << std::endl;
        std::cout << "BLAS faster: x" << d.count() / db.count() << std::endl << std::endl;
    }
}

TEST(matrixd, sqrt_test) {
    {
        auto m = khustup::matrixd<int, 4>{9};
        auto n = m.sqrt();
        for (auto k = 0; k < 4; ++k) {
            ASSERT_EQ(n[k], 3);
        }
    }
    {
        auto m = khustup::matrixd<int, 2, 3, 4>{25};
        auto n = m.sqrt();
        for (auto i = 0; i < 2; ++i) {
            for (auto j = 0; j < 3; ++j) {
                for (auto k = 0; k < 4; ++k) {
                    ASSERT_EQ(n[i][j][k], 5);
                }
            }
        }
    }
}

TEST(matrixd, ownership_test) {
    auto m = khustup::matrixd<int, 2, 3>{5};
    auto n = khustup::matrixd<int, 3, 4>{6};
    auto x{m.dot(n).swap_axes<0, 1>()};
}

TEST(matrixd, different_size_multiplication_test) {
    auto m = khustup::matrixd<float, 1>{-0.25f};
    auto n = khustup::matrixd<float, 3>{-0.4f};
    auto p = m * n;
    for (auto i = 0; i < 3; ++i) {
        ASSERT_TRUE(std::abs(p[i] - 0.1) < 0.0001);
    }
}
