# embed_file(<input> <symbol> <out-var>): turns a file into a C++ source defining
#   extern const unsigned char <symbol>[];  extern const unsigned <symbol>_size;
# so a binary can carry its own images (the splash, the window icon).
function(embed_file input symbol outvar)
  set(out ${CMAKE_BINARY_DIR}/embedded/${symbol}.cpp)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${input})
  if(NOT EXISTS ${out} OR ${input} IS_NEWER_THAN ${out})
    file(READ ${input} hex HEX)
    string(LENGTH "${hex}" n)
    math(EXPR size "${n} / 2")
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex "${hex}")
    string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)" "\\1\n" hex "${hex}")
    file(WRITE ${out} "// generated from ${input} by cmake/embed.cmake\nextern const unsigned char ${symbol}[] = {\n${hex}\n};\nextern const unsigned ${symbol}_size = ${size};\n")
  endif()
  set(${outvar} ${out} PARENT_SCOPE)
endfunction()
