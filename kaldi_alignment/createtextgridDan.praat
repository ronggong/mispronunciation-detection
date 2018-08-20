# Created 3-27-15 Eleanor Chodroff 

wavDir$ = "/Users/ronggong/Documents_using/MTG_document/Jingju_arias"
dir$ = "/Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/splitAli/dan"

Create Strings as file list... list_txt 'dir$'/*.txt
nFiles = Get number of strings

for i from 1 to nFiles
	select Strings list_txt
	filename$ = Get string... i
	basename$ = filename$ - ".txt"
	txtname$ = filename$ - ".txt"


	if startsWith (basename$, "primary_school_recording")

		rind_2017_start = rindex (basename$, "2017")

		subfoldername_len = length (basename$) - rind_2017_start + 1
		# pathname$ = left$ (basename$, rind - 2)
		subfoldername$ = right$ (basename$, subfoldername_len)
		artistname$ = left$ (subfoldername$,  index (subfoldername$, "_") - 1)
		subfoldername2$ = right$ (subfoldername$, length (subfoldername$) - length (artistname$) - 1)

		rind_t = rindex (subfoldername2$, "teacher")
		rind_s = rindex (subfoldername2$, "student")
		if rind_t
			rind = rind_t
		else
			rind = rind_s
		endif
		filename_len = length (subfoldername2$) - rind + 1
		filename_noext$ = right$ (subfoldername2$, filename_len)

		subfoldername2$ = left$ (subfoldername2$, length (subfoldername2$) - filename_len - 1)

		#writeInfoLine: basename$
		#writeInfoLine: pathname$
		#writeInfoLine: subfoldername$
		#writeInfoLine: artistname$
		#writeInfoLine: subfoldername2$
		#writeInfoLine: filename_noext$

		Read from file... 'wavDir$'/primary_school_recording/wav_left/'artistname$'/'subfoldername2$'/'filename_noext$'.wav
	else
		rind_dan = rindex (basename$, "danAll")
		rind_laosheng = rindex (basename$, "laosheng")
		if rind_dan
			rind_roletype_start = rind_dan
			rind_roletype_end = rind_dan + 6
			roletype$ = "danAll"
		else
			rind_roletype_start = rind_laosheng
			rind_roletype_end = rind_laosheng + 8
			roletype$ = "laosheng"
		endif

		filename_len = length (basename$) - rind_roletype_end
		#path_roletype_name = left$ (basename$, rind_roletype_start)
		pathname$ = left$ (basename$,  rind_roletype_start - 2)
		filename_noext$ = right$ (basename$, filename_len)

		#writeInfoLine: rind_roletype_start
		#writeInfoLine: rind_roletype_end
		#writeInfoLine: basename$
		#writeInfoLine: roletype$
		#writeInfoLine: filename_noext$

		Read from file... 'wavDir$'/jingju_a_cappella_singing_dataset/wav_left/'roletype$'/'filename_noext$'.wav
	endif

	dur = Get total duration

	To TextGrid... "kaldiphone"

	#pause 'txtname$'

	select Strings list_txt
	Read Table from tab-separated file... 'dir$'/'txtname$'.txt
	Rename... times
	nRows = Get number of rows
	Sort rows... start
	for j from 1 to nRows
		select Table times
		startutt_col$ = Get column label... 5
		start_col$ = Get column label... 10
		dur_col$ = Get column label... 6
		phone_col$ = Get column label... 7
		if j < nRows
			startnextutt = Get value... j+1 'startutt_col$'
		else
			startnextutt = 0
		endif
		start = Get value... j 'start_col$'
		phone$ = Get value... j 'phone_col$'
		dur = Get value... j 'dur_col$'
		end = start + dur
		select TextGrid 'filename_noext$'
		int = Get interval at time... 1 start+0.005
		if start > 0 & startnextutt = 0
			Insert boundary... 1 start
			Set interval text... 1 int+1 'phone$'
			Insert boundary... 1 end
		elsif start = 0
			Set interval text... 1 int 'phone$'
		elsif start > 0
			Insert boundary... 1 start
			Set interval text... 1 int+1 'phone$'
		endif
		#pause
	endfor
	#pause
	Write to text file... 'dir$'/'basename$'.TextGrid
	select Table times
	plus Sound 'filename_noext$'
	plus TextGrid 'filename_noext$'
	Remove
endfor