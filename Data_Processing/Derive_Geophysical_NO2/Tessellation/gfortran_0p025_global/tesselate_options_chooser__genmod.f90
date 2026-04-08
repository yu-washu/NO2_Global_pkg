        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE TESSELATE_OPTIONS_CHOOSER__genmod
          INTERFACE 
            SUBROUTINE TESSELATE_OPTIONS_CHOOSER(XGRID,YGRID,NXDIM,NYDIM&
     &,ISTART1,ISTART2,ISTART3,ISTART4,JFINIS,ORIENT,DO_GROUP1,         &
     &DOUBLE_FIRST,DOUBLE_LAST,DO_GROUP2,TRIPLE_FIRST,TRIPLE_LAST,      &
     &DOUBLE_MIDDLE)
              INTEGER(KIND=4) :: NYDIM
              INTEGER(KIND=4) :: NXDIM
              REAL(KIND=8) :: XGRID(NXDIM)
              REAL(KIND=8) :: YGRID(NYDIM)
              INTEGER(KIND=4) :: ISTART1
              INTEGER(KIND=4) :: ISTART2
              INTEGER(KIND=4) :: ISTART3
              INTEGER(KIND=4) :: ISTART4
              INTEGER(KIND=4) :: JFINIS
              INTEGER(KIND=4) :: ORIENT
              LOGICAL(KIND=4) :: DO_GROUP1
              LOGICAL(KIND=4) :: DOUBLE_FIRST
              LOGICAL(KIND=4) :: DOUBLE_LAST
              LOGICAL(KIND=4) :: DO_GROUP2
              LOGICAL(KIND=4) :: TRIPLE_FIRST
              LOGICAL(KIND=4) :: TRIPLE_LAST
              LOGICAL(KIND=4) :: DOUBLE_MIDDLE
            END SUBROUTINE TESSELATE_OPTIONS_CHOOSER
          END INTERFACE 
        END MODULE TESSELATE_OPTIONS_CHOOSER__genmod
