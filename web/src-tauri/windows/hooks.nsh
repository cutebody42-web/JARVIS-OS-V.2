!macro NSIS_HOOK_POSTINSTALL
  ${If} ${FileExists} "$INSTDIR\resources\libsodium.dll"
    CopyFiles /SILENT "$INSTDIR\resources\libsodium.dll" "$INSTDIR"
  ${EndIf}
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  Delete "$INSTDIR\libsodium.dll"
!macroend
