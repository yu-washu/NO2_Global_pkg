        !COMPILER-GENERATED INTERFACE MODULE: Mon Jan 28 16:14:31 2019
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE CORNER_AREA__genmod
          INTERFACE 
            FUNCTION CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
              REAL(KIND=8) :: PARS(6,4)
              REAL(KIND=8) :: X1
              REAL(KIND=8) :: X2
              REAL(KIND=8) :: Y1
              REAL(KIND=8) :: Y2
              INTEGER(KIND=4) :: SIDE
              REAL(KIND=8) :: CORNER_AREA
            END FUNCTION CORNER_AREA
          END INTERFACE 
        END MODULE CORNER_AREA__genmod
