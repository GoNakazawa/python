function checkFileSize(btn_id) {
    const size_limit = 1024*1024*10
    const file_input_btn = document.getElementById(btn_id);

    const files = file_input_btn.files;
    for (let i=0;i<files.length;i++)
    {
        if (files[i].size > size_limit){
            alert("ファイルサイズは10MB以下にしてください");
            file_input_btn.value=""
            return false;
        }
    }
}