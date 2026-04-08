        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:32 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE TESSELATE_AREAMASTER__genmod
          INTERFACE 
            SUBROUTINE TESSELATE_AREAMASTER(XGRID,YGRID,NXDIM,NYDIM,    &
     &CORNER_COORDS,SIDE_PARAMS,ORIENT,ISTART1,ISTART2,ISTART3,ISTART4, &
     &JFINIS,AREA,SUM,YLIMIT_LOWER,YLIMIT_UPPER)
              INTEGER(KIND=4) :: NYDIM
              INTEGER(KIND=4) :: NXDIM
              REAL(KIND=8) :: XGRID(NXDIM)
              REAL(KIND=8) :: YGRID(NYDIM)
              REAL(KIND=8) :: CORNER_COORDS(4,2)
              REAL(KIND=8) :: SIDE_PARAMS(6,4)
              INTEGER(KIND=4) :: ORIENT
              INTEGER(KIND=4) :: ISTART1
              INTEGER(KIND=4) :: ISTART2
              INTEGER(KIND=4) :: ISTART3
              INTEGER(KIND=4) :: ISTART4
              INTEGER(KIND=4) :: JFINIS
              REAL(KIND=8) :: AREA(NYDIM,NXDIM)
              REAL(KIND=8) :: SUM
              INTEGER(KIND=4) :: YLIMIT_LOWER(NXDIM)
              INTEGER(KIND=4) :: YLIMIT_UPPER(NXDIM)
            END SUBROUTINE TESSELATE_AREAMASTER
          END INTERFACE 
        END MODULE TESSELATE_AREAMASTER__genmod
