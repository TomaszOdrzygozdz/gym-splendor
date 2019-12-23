# _(Not only)_ Coding standards

* **README**

    Here you can find how to write README:
    * [Making READMEs readable](https://open-source-guide.18f.gov/making-readmes-readable/)
    * [README-Template.md by Billie Thompson](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)


* **Python**

    [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) and [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) are in operation!

    If you are emacs user, I recommend installing this package: py-autopep8. Configuration:  
    ```elisp
    ;; enable autopep8 formatting on save
    (require 'py-autopep8)
    (add-hook 'elpy-mode-hook 'py-autopep8-enable-on-save)
    ```
    If you look for the best python/markdown/everything IDE and want to configure it easily, here is a guide for you: https://realpython.com/blog/python/emacs-the-best-python-editor/ and then http://jblevins.org/projects/markdown-mode/ .


* **Git commits**

    * [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) is in operation.

    * Remote branch names should follow those templates:

        * Personal branches: `<user name>/<your branch name>`
          These keep developer changes that are actively developed before merging into the master or one of develop branches.
        * Develop branches: `dev/<branch name e.g. r0.0.2>`
          These keep development code before release and merge into the master.


* **Merge requests**

    * If you want to commit to the master branch or one of develop branches **create a Merge Request**. Code-review and acceptance of at least one maintainer is mandatory.
