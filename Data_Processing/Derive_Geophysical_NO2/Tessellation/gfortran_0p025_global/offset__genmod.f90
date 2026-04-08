        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE OFFSET__genmod
          INTERFACE 
            FUNCTION OFFSET(START,GRID,NGRID,ASCENDING,VALUE)
              INTEGER(KIND=4) :: NGRID
              INTEGER(KIND=4) :: START
              REAL(KIND=8) :: GRID(NGRID)
              INTEGER(KIND=4) :: ASCENDING
              REAL(KIND=8) :: VALUE
              INTEGER(KIND=4) :: OFFSET
            END FUNCTION OFFSET
          END INTERFACE 
        END MODULE OFFSET__genmod
