        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:32 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE TESSELATE_OPTIONS_2__genmod
          INTERFACE 
            SUBROUTINE TESSELATE_OPTIONS_2(XGRID,YGRID,NXDIM,NYDIM,CC,  &
     &ORIENT,TRIPLE_FIRST,TRIPLE_LAST,DOUBLE_MIDDLE,PARS,ISTART1,ISTART2&
     &,ISTART3,ISTART4,JFINIS,AREA,YLIMIT_LOWER,YLIMIT_UPPER)
              INTEGER(KIND=4) :: NYDIM
              INTEGER(KIND=4) :: NXDIM
              REAL(KIND=8) :: XGRID(NXDIM)
              REAL(KIND=8) :: YGRID(NYDIM)
              REAL(KIND=8) :: CC(4,2)
              INTEGER(KIND=4) :: ORIENT
              LOGICAL(KIND=4) :: TRIPLE_FIRST
              LOGICAL(KIND=4) :: TRIPLE_LAST
              LOGICAL(KIND=4) :: DOUBLE_MIDDLE
              REAL(KIND=8) :: PARS(6,4)
              INTEGER(KIND=4) :: ISTART1
              INTEGER(KIND=4) :: ISTART2
              INTEGER(KIND=4) :: ISTART3
              INTEGER(KIND=4) :: ISTART4
              INTEGER(KIND=4) :: JFINIS
              REAL(KIND=8) :: AREA(NYDIM,NXDIM)
              INTEGER(KIND=4) :: YLIMIT_LOWER(NXDIM)
              INTEGER(KIND=4) :: YLIMIT_UPPER(NXDIM)
            END SUBROUTINE TESSELATE_OPTIONS_2
          END INTERFACE 
        END MODULE TESSELATE_OPTIONS_2__genmod
