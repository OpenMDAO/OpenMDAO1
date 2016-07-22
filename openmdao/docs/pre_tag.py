import sys, os, shutil, re

def setup_dirs():
  dir = os.path.dirname(__file__)
  tagdir = os.path.join(dir, "tags")

  if os.path.isdir(tagdir):
     shutil.rmtree(tagdir)

  if not os.path.isdir(tagdir):
    os.mkdir(tagdir)

  return tagdir

def find_tags(docdirs, tagdir):
  for docdir in docdirs:
    for dirpath, dirnames, filenames in os.walk(docdir):
      for filename in filenames:
        #the path to the file being read for tags
        sourcefile = os.path.join(dirpath, filename)
        #a file object for the file being read for tags
        textfile = open( sourcefile, 'r')
        #the text of the entire sourcefile
        filetext = textfile.read()
        textfile.close()

        #pull all tag directives out of the filetext
        matches = re.findall(".. tags::.*$", filetext)

        #for every instance of tag directive, get a list of tags
        for match in matches:
          match=match.lstrip(".. tags::")
          taglist=match.split(", ")

          #for every tag noted, we have to do two things:
          #1.
          for tag in taglist:
            filepath = os.path.join(tagdir, (tag+".rst"))

            #if the tagfile doesn't exist, let's put in a header
            #the first time through
            if not os.path.exists(filepath):
              tagfileheader="""
===============
%s
===============

  .. toctree::
     :maxdepth: 1

""" % tag

              #write in the header for this tag's file.
              with open(filepath, 'a') as tagfile:
                tagfile.write(tagfileheader)

            #write a link into the appropriate tagfile.
            #the link is to the document in which the tag appears.
            with open(filepath, 'a') as tagfile:
              #tagfile.write("     .. _%s: ../%s\n" % (filename, sourcefile))
              tagfile.write("     ../%s\n" % (sourcefile))
            #write the links to the tagfile in this rst file
            #this link is the tagname as a hyperlink to the tag's file
            #with open(sourcefile, 'a') as source:
            #  source.write(".. _%s ./%s\n" % (tag, filepath))


def main(args=None):
    """
    process command line arguments and perform requested task
    """
    if args is None:
        args = sys.argv[1:]

    tagdir = setup_dirs()

    docdirs=['conversion-guide', 'getting-started', 'usr-guide']

    find_tags(docdirs, tagdir)




if __name__ == '__main__':
    sys.exit(main())
