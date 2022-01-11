" GENERAL ------------------------------------------------------------- {{{ 
" Disable compatibility with vi which can cause unexpected issues
set nocompatible    

" Enable filetype detection. Vim will try to autodetect the file
filetype off

" Enable plugins and load plugin for detected filetype
filetype plugin indent on

" Load an indent file for the detected filetype
filetype indent on

set noswapfile

" UTF-8 support
set encoding=utf-8

" Access system keyboard
set clipboard=unnamed

" If Vim version is equal to or greater than 7.3 enable undofile.
" This allows you to undo changes to a file even after saving it.
if version >= 703
    set undodir=~/.vim/backup
    set undofile
    set undoreload=10000
endif

" YouCompleteMe setting
" Use homebrew's clangd
let g:ycm_clangd_binary_path = trim(system('brew --prefix llvm')).'/bin/clangd'

" Makes autocomplete window go away after completion
let g:ycm_autoclose_preview_window_after_completion=1

" Spell checking in UK-English
set spell spelllang=en_gb
" }}}

" PLUGINS ------------------------------------------------------------- {{{ 
" set runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" Let Vundle manage itself
Plugin 'VundleVim/Vundle.vim'

" Plugins need to be added here
Plugin 'morhetz/gruvbox'                                            " Retro color scheme
Plugin 'Lokaltog/powerline', {'rtp': 'powerline/bindings/vim/'}     " Powerline
Plugin 'tpope/vim-fugitive'                                         " Git support for vim
Plugin 'lervag/vimtex'                                              " LateX support for vim
Plugin 'vim-autoformat/vim-autoformat'                              " Provides autoformatting for VimTex
Plugin 'ervandew/supertab'                                          " Makes <Tab> do inserting
Plugin 'dense-analysis/ale'                                         " Syntax checker various languages
Plugin 'preservim/nerdtree'                                         " File system explorer for vim
Plugin 'tmhedberg/SimpylFold'                                       " Better folding for coding
Plugin 'vim-scripts/indentpython.vim'                               " Indent checks for python
Plugin 'nvie/vim-flake8'                                            " Pep8 support for python
Plugin 'Valloric/YouCompleteMe'                                     " Autocomplete for many languages

" Calls the plugins
call vundle#end()           
" }}}

" VISUAL ------------------------------------------------------------- {{{ 
" General setting
set number

" Console colors for python
autocmd vimenter * ++nested colorscheme gruvbox

" Sets the background to dark
set bg=dark

" Rectify the colorscheme (also in BASH-export)
let g:solarized_termcolors=256
set t_Co=256

" Gruvbox settings
let g:gruvbox_bold ='0'
let g:gruvbox_italic = '0'

" Highlight cursor lines like cross
set cursorline
set cursorcolumn

" Display cursorline and cursorcolumn ONLY in active window.
augroup cursor_off
    autocmd!
    autocmd WinLeave * set nocursorline nocursorcolumn
    autocmd WinEnter * set cursorline cursorcolumn
augroup END

" Python
let python_highlight_all=1

" C++

" Turn on syntax highlighting
syntax on	
" }}}

" SEARCH ------------------------------------------------------------- {{{ 
" Make the search better
set hlsearch
set ignorecase	
set smartcase
set incsearch
set history=1000
" }}}

" PYTHON ------------------------------------------------------------- {{{ 
" Python tab-setting for vim
set tabstop=4
set softtabstop=4
set shiftwidth=4
set textwidth=79
set smarttab
set expandtab
set autoindent
set fileformat=unix

" General Layout for Python programming
set splitbelow

" }}}

" LATEX ------------------------------------------------------------- {{{
" Compiler options
let g:vimtex_compiler_method='latexmk'
let g:vimtex_compiler_latexmk={
    \ 'background' : 1,
    \ 'build_dir' : 'build',
    \ 'callback' : 1,
    \ 'continuous' : 1,
    \ 'executable' : 'latexmk',
    \ 'options' : [
    \   '-pdf',
    \   '-verbose',
    \   '-file-line-error',
    \   '-synctex=1',
    \   '-interaction=nonstopmode',
    \ ],
    \}

" Viewer options
let g:vimtex_view_method='zathura'

" Latex flavor
let g:tex_flavor='latex'

" Change the localleader for VimTex
let maplocalleader=',' 
" }}}

" FILE EXPLORERS------------------------------------------------------------- {{{ 
" Enable autocompleting after pressing tab for wildmenu"
set wildmenu

" Make wildmenu behave like Bash
set wildmode=list:longest

" Make wildmenu ignore files not meant for vim editing
set wildignore=*.docx,*.jpg,*.png,*.gif,*.pdf,*.pyc,*.exe,*.flv,*.img,.xlsx

" Let NERDtree ignore certain files and directories
let NERDTreeIgnore=['\.git$','\.jpg$','\.mp4$','\.ogg$','\.iso$','\.pdf$','\.pyc$','\.odt$','\.png$','\.gif$','\.db$']
" }}}

" NAVIGATION ------------------------------------------------------------- {{{ 
" This will enable code folding with the marker method
augroup filetype_vim
    autocmd!
    autocmd FileType vim setlocal foldmethod=marker
augroup end
"
" Enable code folding
set foldmethod=indent
set foldlevel=99

" Shows docstrings for folded code
let g:SimpylFold_docstring_preview=1
" }}}

" REMAPS ------------------------------------------------------------- {{{ 
inoremap jj <esc>
nnoremap o o<esc>
nnoremap O O<esc>

" Yank from cursor to the end of line
nnoremap Y y$

" Enable folding with spacebar
nnoremap <space> za

" Split navigation
nnoremap <C-J> <C-W><C-J>
nnoremap <C-K> <C-W><C-K>
nnoremap <C-L> <C-W><C-L>
nnoremap <C-H> <C-W><C-H>

" NERDTree mappings
nnoremap <f3> :NERDTreeToggle<CR>

" LateX preview mappings
nnoremap <f4> :LLPStartPreview<CR>

" YouCompleteMe mapping for goto definition
map <leader>g :YouCompleter GoToDefinitionElseDeclaration<CR>
" }}}

" VIMSCRIPTS ------------------------------------------------------------- {{{ 
" Run python script in Vim. <CR> (carriage return), :!clear, clears screen
" command
nnoremap <f5> :w <CR>:!clear <CR>:!python3 % <CR>

" Adds virtulenv support for python
" py << EOF
" import os
" import sys
" if 'VIRTUAL_ENV' in os.environ:
"     project_base_dir = os.environ['VIRTUAL_ENV']
"     activate_this = os.path.join(project_base_dir, 'bin/activate_this.py')
"     execfile(activate_this, dict(__file__=activate_this))
" EOF
" }}}

" STATUS LINE ------------------------------------------------------------- {{{ 
" Clear status line when vimrc is reloaded.
set statusline=

" Status line left side.
set statusline+=\ %F\ %M\ %Y\ %R

" Use a divider to separate the left side from the right side.
set statusline+=%=

" Status line right side.
set statusline+=\ ascii:\ %b\ hex:\ 0x%B\ row:\ %l\ col:\ %c\ percent:\ %p%%

" Show the status on the second to last line.
set laststatus=2
" }}}
